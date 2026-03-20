# Plugin Development Guide

This guide explains how to implement production plugins beyond the quick demonstration.

See also: [PLUGIN_DEMONSTRATION](PLUGIN_DEMONSTRATION.md)

## 1. Contract summary

A plugin module should provide:

1. `PLUGIN_MANIFEST` (or `get_plugin_manifest()`)
2. one of:
   - `register_plugin(registrar)`
   - `PLUGIN_METHODS`

Required manifest fields:

- `plugin_id`
- `name`
- `version`
- `api_version` (currently `"1"`)

Optional fields:

- `description`
- `author`
- `capabilities`
- `entrypoint`

## 2. Feature key targeting

Methods are registered to a feature key + method name.

Example:

```python
registrar.register_method("atlas_registration.registration", "brainglobe", my_func)
```

If selected method exists as plugin method, it overrides same-name builtin method at runtime.

## 3. Method signature strategy

Use the builtin feature function signature as reference for compatibility.

Examples:

- batch auto typing method receives `(folder, options, config)`
- simplification method receives `(dataframe, config)`
- validation run method receives `(swc_text, config)`

When replacing a method, return the same output shape expected by caller.

## 4. Loading behavior

- `swctools plugins load <module>` is process-local.
- `SWCTOOLS_PLUGINS` enables autoload each new process.

```bash
export SWCTOOLS_PLUGINS="my_lab_plugins.plugin_a,my_lab_plugins.plugin_b"
```

## 5. Safety recommendations

- validate inputs early (paths/config fields)
- return deterministic payloads
- include clear error messages
- avoid side effects outside output/log paths
- write logs/reports for auditability

## 6. Versioning recommendations

- bump plugin `version` on behavior changes
- keep `api_version` aligned with app support
- maintain changelog in plugin repo/lab docs

## 7. Testing checklist

1. module import works in target venv
2. `swctools plugins load ...` succeeds
3. `swctools plugins list --feature-key ...` shows method
4. command using `--config-json {"method":"..."}` executes
5. output/report files are generated correctly

## 8. Thin-wrapper pattern

For third-party tools (example: BrainGlobe CLI wrappers):

- plugin method accepts swctools inputs
- plugin method calls external command/library
- plugin method maps output back into swctools result payload

This pattern avoids rewriting third-party algorithms while still integrating into SWC-Studio workflows.
