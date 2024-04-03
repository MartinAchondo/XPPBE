import os
import platform
import shutil


def generate_msms_mesh(mesh_xyzr_path, output_dir, output_name, density, probe_radius=1.4):

    path = os.path.join(output_dir, f'{output_name}_d{density}')
    msms_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Mesh_softwares', "MSMS", "")
    if platform.system() == "Linux":
        external_file = "msms"
        os.system("chmod +x " + msms_dir + external_file)
    elif platform.system() == "Windows":
        external_file = "msms.exe"
    command = (
        msms_dir
        + external_file
        + " -if "
        + mesh_xyzr_path
        + " -of "
        + path
        + " -p "
        + str(probe_radius)
        + " -d "
        + str(density)
        + " -no_header"
    )
    print(command)
    os.system(command)


def generate_nanoshaper_mesh(
    mesh_xyzr_path,
    output_dir,
    output_name,
    density,
    probe_radius=1.4,
    save_mesh_build_files=True,
):

    nanoshaper_dir = os.path.join(
        'code','Model','Mesh','Mesh_softwares', "NanoShaper", ""
    )
    nanoshaper_temp_dir = os.path.join(output_dir, "nanotemp", "")

    if not os.path.exists(nanoshaper_temp_dir):
        os.makedirs(nanoshaper_temp_dir)

    # Execute NanoShaper
    config_template_file = open(nanoshaper_dir + "config", "r")
    config_file = open(nanoshaper_temp_dir + "surfaceConfiguration.prm", "w")
    for line in config_template_file:
        if "XYZR_FileName" in line:
            line = "XYZR_FileName = " + mesh_xyzr_path + " \n"
        elif "Grid_scale" in line:
            line = "Grid_scale = {:04.1f} \n".format(density)
        elif "Probe_Radius" in line:
            line = "Probe_Radius = {:03.1f} \n".format(probe_radius)

        config_file.write(line)

    config_file.close()
    config_template_file.close()

    os.chdir(nanoshaper_temp_dir)
    if platform.system() == "Linux":
        os.system("chmod +x " + nanoshaper_dir + "NanoShaper")
        os.system(nanoshaper_dir + "NanoShaper")
    elif platform.system() == "Windows":
        if platform.architecture()[0] == "32bit":
            os.system(
                nanoshaper_dir
                + "NanoShaper32.exe"
                + " "
                + nanoshaper_temp_dir
                + "surfaceConfiguration.prm"
            )
        elif platform.architecture()[0] == "64bit":
            os.system(
                nanoshaper_dir
                + "NanoShaper64.exe"
                + " "
                + nanoshaper_temp_dir
                + "surfaceConfiguration.prm"
            )
    os.chdir("..")

    try:
        vert_file = open(nanoshaper_temp_dir + "triangulatedSurf.vert", "r")
        vert = vert_file.readlines()
        vert_file.close()
        face_file = open(nanoshaper_temp_dir + "triangulatedSurf.face", "r")
        face = face_file.readlines()
        face_file.close()

        vert_file = open(output_name + ".vert", "w")
        vert_file.write("".join(vert[3:]))
        vert_file.close()
        face_file = open(output_name + ".face", "w")
        face_file.write("".join(face[3:]))
        face_file.close()

        if not save_mesh_build_files:
            shutil.rmtree(nanoshaper_temp_dir)

        os.chdir("..")

    except (OSError, FileNotFoundError):
        print("The file doesn't exist or it wasn't created by NanoShaper")

