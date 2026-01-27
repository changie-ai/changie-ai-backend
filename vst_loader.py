import os
import sounddevice as sd
import numpy as np
import vst  # this should match whatever module you were using earlier for VST processing

def load_vst(plugin_name):
    """Loads a VST plugin from the plugins folder."""
    vst_path = os.path.join("plugins", f"{plugin_name}.vst")
    if not os.path.exists(vst_path):
        raise FileNotFoundError(f"VST plugin not found: {vst_path}")
    return vst.VSTHost(vst_path)
