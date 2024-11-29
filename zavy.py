import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="zavychromaxl_v40.safetensors"
        )

        deepcache = NODE_CLASS_MAPPINGS["DeepCache"]()
        deepcache_109 = deepcache.apply(
            cache_interval=3,
            cache_depth=5,
            start_step=0,
            end_step=1000,
            model=get_value_at_index(checkpointloadersimple_4, 0),
        )

        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        freeu_v2_92 = freeu_v2.patch(
            b1=1.1, b2=1.2, s1=0.6, s2=0.4, model=get_value_at_index(deepcache_109, 0)
        )

        lorastackloader_pop = NODE_CLASS_MAPPINGS["LoraStackLoader_PoP"]()
        lorastackloader_pop_55 = lorastackloader_pop.apply_loras(
            switch_1="On",
            lora_name_1="xl_more_art-full_v1.safetensors",
            strength_model_1=0.2,
            strength_clip_1=0.2,
            switch_2="On",
            lora_name_2="extremely detailed.safetensors",
            strength_model_2=0.2,
            strength_clip_2=0.2,
            switch_3="Off",
            lora_name_3="None",
            strength_model_3=1,
            strength_clip_3=1,
            model=get_value_at_index(freeu_v2_92, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cr_aspect_ratio = NODE_CLASS_MAPPINGS["CR Aspect Ratio"]()
        cr_aspect_ratio_100 = cr_aspect_ratio.Aspect_Ratio(
            width=1320,
            height=1024,
            aspect_ratio="custom",
            swap_dimensions="Off",
            upscale_factor=1,
            prescale_factor=1,
            batch_size=2,
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_102 = cliptextencode.encode(
            text="Beautiful woman in blonde",
            clip=get_value_at_index(lorastackloader_pop_55, 1),
        )

        cliptextencode_105 = cliptextencode.encode(
            text="bad face, bad hand, bad legs, bad lips, bad eyes, bad nose, distortion, merged people, exposed genital body parts, exposed breasts, exposed boobs, nudity, cleavage, bikini, underwear, swimwear, half exposed boobs, half exposed breast, extra fingers, missing fingers, missing leg, extra legs, mutation, bad anatomy, nighty",
            clip=get_value_at_index(lorastackloader_pop_55, 1),
        )

        ksampleradvanced = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            ksampleradvanced_65 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=random.randint(1, 2**64),
                steps=18,
                cfg=3,
                sampler_name="uni_pc",
                scheduler="karras",
                start_at_step=0,
                end_at_step=10000,
                return_with_leftover_noise="enable",
                model=get_value_at_index(lorastackloader_pop_55, 0),
                positive=get_value_at_index(cliptextencode_102, 0),
                negative=get_value_at_index(cliptextencode_105, 0),
                latent_image=get_value_at_index(cr_aspect_ratio_100, 5),
            )

            vaedecode_45 = vaedecode.decode(
                samples=get_value_at_index(ksampleradvanced_65, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            saveimage_114 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_45, 0)
            )


if __name__ == "__main__":
    main()
