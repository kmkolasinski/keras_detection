import weakref
import tensorflow as tf
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.util import tf_decorator


keras = tf.keras


def from_keras_model(model: keras.Model) -> tf.lite.TFLiteConverter:
    """Creates a TFLiteConverter object from a Keras model.

    Args:
      model: tf.Keras.Model

    Returns:
      TFLiteConverter object.
    """
    input_signature = None
    # If the model's call is not a `tf.function`, then we need to first get its
    # input signature from `model_input_signature` method. We can't directly
    # call `trace_model_call` because otherwise the batch dimension is set
    # to None.
    # Once we have better support for dynamic shapes, we can remove this.
    if not isinstance(model.call, _def_function.Function):
        # Pass `keep_original_batch_size=True` will ensure that we get an input
        # signature including the batch dimension specified by the user.
        input_signature = _saving_utils.model_input_signature(
            model, keep_original_batch_size=True
        )

    func = _saving_utils.trace_model_call(model, input_signature)
    func._defun_with_scope = lambda s: _defun_with_scope(func, s)
    concrete_func = func.get_concrete_function()
    return tf.lite.TFLiteConverter([concrete_func])


def _defun_with_scope(self, scope):
    """Creates a defun wrapped inside a variable creator scope."""

    weak_wrapped_fn = None

    def wrapped_fn(*args, **kwds):
        """Wraps `self._python_function` in a variable creator scope."""
        with ops.get_default_graph()._variable_creator_scope(scope, priority=50):
            return weak_wrapped_fn().__wrapped__(*args, **kwds)

    weak_wrapped_fn = weakref.ref(wrapped_fn)

    fun = self._defun(tf_decorator.make_decorator(self._python_function, wrapped_fn))

    fun._create_graph_function = lambda args, kwargs: _create_graph_function(
        fun, args, kwargs
    )
    return fun


def _create_graph_function(self, args, kwargs, override_flat_arg_shapes=None):
    """Create a `ConcreteFunction` from `args` and `kwargs`."""

    self.tracing_count += 1
    if self.input_signature is None:
        arglen = len(args)
    else:
        arglen = len(self.input_signature)
    base_arg_names = self._function_spec.arg_names[:arglen]
    num_missing_args = arglen - len(self._function_spec.arg_names)
    missing_arg_names = [self._function_spec.vararg_name] * num_missing_args
    # Produce a list of missing args of the form ["arg_0", "arg_1", ...],
    # where arg is based on the self._function_spec.vararg_name.
    missing_arg_names = ["%s_%d" % (arg, i) for i, arg in enumerate(missing_arg_names)]
    arg_names = base_arg_names + missing_arg_names

    graph_function = _function.ConcreteFunction(
        func_graph_module.func_graph_from_py_func(
            self._name,
            self._python_function,
            args,
            kwargs,
            self.input_signature,
            autograph=self._autograph,
            autograph_options=self._autograph_options,
            arg_names=arg_names,
            override_flat_arg_shapes=override_flat_arg_shapes,
            capture_by_value=self._capture_by_value,
            add_control_dependencies=False,
        ),
        self._function_attributes,
        # Tell the ConcreteFunction to clean up its graph once it goes out of
        # scope. This is not the default behavior since it gets used in some
        # places (like Keras) where the FuncGraph lives longer than the
        # ConcreteFunction.
        shared_func_graph=False,
    )
    return graph_function
