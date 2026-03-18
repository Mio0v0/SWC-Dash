"""Public library API for swctools.

All interfaces (GUI, CLI, and external Python callers) should import these
functions or the corresponding tool/feature modules.
"""

from __future__ import annotations

from swctools.core.auto_typing import RuleBatchOptions
from swctools.plugins import (
    autoload_plugins_from_environment,
    get_plugin,
    list_all_feature_methods,
    list_feature_methods,
    list_plugins,
    load_plugin_module,
    load_plugins,
    register_method,
    register_plugin_manifest,
    register_plugin_method,
    unregister_method,
    unregister_plugin,
)
from swctools.tools.analysis.features.summary import analyze_file as analysis_summary_file
from swctools.tools.atlas_registration.features.registration import register_to_atlas
from swctools.tools.batch_processing.features.auto_typing import run_folder as batch_auto_typing
from swctools.tools.batch_processing.features.batch_validation import (
    validate_folder as batch_validate_folder,
)
from swctools.tools.batch_processing.features.radii_cleaning import (
    clean_folder as batch_radii_cleaning,
    clean_path as radii_clean_path,
)
from swctools.tools.batch_processing.features.swc_splitter import (
    split_folder as batch_split_folder,
)
from swctools.tools.morphology_editing.features.dendrogram_editing import (
    reassign_subtree_types,
    reassign_subtree_types_in_file,
)
from swctools.tools.morphology_editing.features.simplification import (
    simplify_file as morphology_smart_decimation_file,
    simplify_swc_text as morphology_smart_decimation_text,
)
from swctools.tools.validation.features.auto_fix import auto_fix_file, auto_fix_text
from swctools.tools.validation.features.auto_typing import (
    run_file as validation_auto_typing_file,
)
from swctools.tools.validation.features.run_checks import (
    validate_file as validation_run_file,
    validate_text as validation_run_text,
)
from swctools.tools.visualization.features.mesh_editing import (
    build_mesh_from_file,
    build_mesh_from_text,
)

__all__ = [
    "RuleBatchOptions",
    "batch_validate_folder",
    "batch_split_folder",
    "batch_auto_typing",
    "batch_radii_cleaning",
    "radii_clean_path",
    "auto_fix_text",
    "auto_fix_file",
    "validation_auto_typing_file",
    "validation_run_text",
    "validation_run_file",
    "build_mesh_from_text",
    "build_mesh_from_file",
    "reassign_subtree_types",
    "reassign_subtree_types_in_file",
    "morphology_smart_decimation_text",
    "morphology_smart_decimation_file",
    "register_to_atlas",
    "analysis_summary_file",
    "load_plugin_module",
    "load_plugins",
    "autoload_plugins_from_environment",
    "register_plugin_manifest",
    "register_plugin_method",
    "unregister_plugin",
    "list_plugins",
    "get_plugin",
    "register_method",
    "unregister_method",
    "list_feature_methods",
    "list_all_feature_methods",
]
