import os
import numpy as np
import subprocess
import re
import itertools
# material, pseudopotential setting {material : (upf_files , pp_type)}
pseudopotential_type = ["SG15", "TM"][0]
materials_pseudopotentials = {
    "diamond" : (None, pseudopotential_type),
#    "silicon" : (None, pseudopotential_type),
#    "Fe_fcc"  : (None, pseudopotential_type),
#    "CaTiO3"  : (None, pseudopotential_type),
#    "MAPbI3"  : (None, pseudopotential_type),
#    "CsPbI3"  : (None, pseudopotential_type),
#    "C60"     : (None, pseudopotential_type),
#    "ATP"     : (None, pseudopotential_type),
#    "GSH"     : (None, pseudopotential_type),
#    "aspirin" : (None, pseudopotential_type),
}
#setting spacing 
spacing_supercell_nbands_list = { 
"diamond" : {"spacing" : (0.2, ), "supercell" : ([1, 1, 1], ), "nbands" : (20, )},
# "diamond" : {"spacing" : (0.2, ), "supercell" : ([3, 3, 3], ), "nbands" : (20, )},
#"silicon" : {"spacing" : (0.2, ), "supercell" : ([4, 2, 2], ), "nbands" : (20, )},
#"Fe_fcc"  : {"spacing" : (0.2, ), "supercell" : ([2, 2, 2], ), "nbands" : (20, )},
#"CaTiO3"  : {"spacing" : (0.2, ), "supercell" : ([2, 2, 2], ), "nbands" : (20, )},
#"MAPbI3"  : {"spacing" : (0.2, ), "supercell" : ([2, 2, 1], ), "nbands" : (100, )},
#"CsPbI3"  : {"spacing" : (0.2, ), "supercell" : ([2, 2, 2], ), "nbands" : (20, )},
#"C60"     : {"spacing" : (0.5, ), "supercell" : ([1, 1, 1], ), "nbands" : (150, )},
#"ATP"     : {"spacing" : (0.5, ), "supercell" : ([1, 1, 1], ), "nbands" : (100, )},
#"GSH"     : {"spacing" : (0.5, ), "supercell" : ([1, 1, 1], ), "nbands" : (70, )},
#"aspirin" : {"spacing" : (0.5, ), "supercell" : ([1, 1, 1], ), "nbands" : (40, )},
}

# location that cif,sdf files's diractory of material 
cif_file_path = "./"

#calculation script setting 
# script = "run_files_test.py"
script = "run_files.py"

# preconditioner setting
preconditioner_solve = {
        # "general" : (("gapp", "poisson"), (None,)),
        "general" : (("gapp",), (None,)),
        "ISI"     : (("shift-and-invert",),("gapp", "Neumann")),
        "Neumann" : (("Neumann",),(None,)),
}

# phase = ["scf", "fixed"][0] ##setting phase 
phase = ["scf", "fixed"][1] ##setting phase 
gapp_pcg_num = 5
Neumann_pcg_num = 2

# settinf Neumann orders 
innerorder_list = [
        #1,
        #2,
        10
        ]
outerorder_list = [
#                   0,  
#                   1,
#                   3,
#                   5,
#                   7,
#                   9,
#                   11,
#                   13,
#                   20,
#                   35,
#                   50,
                    "dynamic"

                   ]

error_cutoff_list = [
        -0.1,
        -0.2,
        -0.3,
        -0.4,
        -0.5,
        -0.6,
        -0.7,
        ]

# 결과 저장 폴더
output_dir = "test_result"
os.makedirs(output_dir, exist_ok=True)

def add_opt(cmd: list[str], flag: str, value):
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend([str(v) for v in value])
    else:
        cmd.extend([flag, str(value)])

for precond_type, (precond_list, innerprecond_list) in preconditioner_solve.items():
    for precond, innerprecond in itertools.product(precond_list, innerprecond_list):
        for material, (upf_files, pp_type) in materials_pseudopotentials.items():
            # density_file = f"density_{material}.pt"  # density files name

            if precond == "gapp": 
                summary_file = os.path.join(output_dir, f"results_summary_gapp_{material}.txt")
            
            elif precond == "poisson":
                summary_file = os.path.join(output_dir, f"results_summary_poisson_{material}.txt")

            elif precond == "shift-and-invert" and innerprecond == "gapp":
                summary_file = os.path.join(output_dir, f"results_summary_ISI_gapp_pcg_{gapp_pcg_num}_{material}.txt")

            elif precond == "shift-and-invert" and innerprecond == "Neumann":
                summary_file = os.path.join(output_dir, f"results_summary_shift_Neumann_pcg_{Neumann_pcg_num}_{material}.txt")
            elif precond == "Neumann":
                summary_file = os.path.join(output_dir, f"results_summary_Neumann_{material}.txt")
            else:
                raise NotImplementedError(f"Unknown precond: {precond}, inner: {innerprecond}")



            if not os.path.exists(summary_file):
                with open(summary_file, "w") as f:
                    if precond == "Neumann":
                        f.write("Spacing, Preconditioner, Material, NBands, Supercell, Median Elapsed Time (s), Max Iterations, Total Precond Time, Per Precond Time, Solver Type, Order, PCG_Number\n")

                    elif precond == "shift-and-invert" and innerprecond == "Neumann":
                        f.write("Spacing, Preconditioner, InnerPreconditioner, Material, NBands, Supercell, Median Elapsed Time (s), Max Iterations, Total Precond Time, Per Precond Time, Solver Type, Order, PCG_Number\n")

                    elif precond == "shift-and-invert" and innerprecond == "gapp":
                        f.write("Spacing, Preconditioner, InnerPreconditioner, Material, NBands, Supercell, Median Elapsed Time (s), Max Iterations, Total Precond Time, Per Precond Time, Solver Type, PCG_Number\n")

                    else:
                        f.write("Spacing, Preconditioner, Material, NBands, Supercell, Median Elapsed Time (s), Max Iterations, Total Precond Time, Per Precond Time, Solver Type,\n")


            env = os.environ.copy()
            env["MKL_SERVICE_FORCE_INTEL"] = "1"
            env["MKL_THREADING_LAYER"] = "GNU" 
            spacing_list = spacing_supercell_nbands_list[material]["spacing"]
            supercell_factors_list = spacing_supercell_nbands_list[material]["supercell"]
            nbands_list = spacing_supercell_nbands_list[material]["nbands"]

            base_combos = itertools.product(nbands_list, spacing_list, (precond,), supercell_factors_list)  
            if precond == "Neumann":
                combos = (
                (nbands, spacing, precond, supercell, outerorder, error_cutoff, None, None) for nbands, spacing, precond, supercell, outerorder, error_cutoff in itertools.product(nbands_list, spacing_list, (precond,), supercell_factors_list, outerorder_list, error_cutoff_list)
                )
                gapp_pcg_num = None
                Neumann_pcg_num = None

            elif precond in ("gapp", "poisson"):
                combos = (
                (nbands, spacing, precond, supercell, None, None, None, None) for nbands, spacing, precond, supercell in base_combos)
                gapp_pcg_num = None
                Neumann_pcg_num = None

            elif precond == "shift-and-invert" and innerprecond == "gapp":
                combos = (
                (nbands, spacing, precond, supercell, None, None, "gapp", None) for nbands, spacing, precond, supercell in base_combos)
                gapp_pcg_num = 5
                Neumann_pcg_num = None

            elif precond == "shift-and-invert" and innerprecond == "Neumann":
                combos = (
                (nbands, spacing, precond, supercell, None, None, "Neumann", innerorder) for nbands, spacing, precond, supercell, innerorder in itertools.product(nbands_list, spacing_list, (precond,), supercell_factors_list, innerorder_list)
                )
                gapp_pcg_num = None
                Neumann_pcg_num = 2

            else:
                raise NotImplementedError(f"Unknown precond: {precond}, inner: {innerprecond}")

            for nbands, i, precond_type, supercell_factors, outerorder, error_cutoff, innerprecond, innerorder in combos:
                diagonal_total_times = []
                total_precond_times = []
                max_diagonal_iterations_list = []
                max_scf_iteration_list = []
                results = []

                adjusted_nbands = nbands * np.prod(supercell_factors)
                if precond in ("gapp", "poisson"):
                    folder_path = os.path.join(
                        output_dir,
                        material,
                        f"nbands_{adjusted_nbands}",
                        f"spacing_{i}",
                        f"supercell_{'_'.join(map(str, supercell_factors))}",
                        f"precond_{precond}_{innerprecond}",
                        f"phase_{phase}",
                        
                    )
                    os.makedirs(folder_path, exist_ok=True)

                elif precond == "shift-and-invert" and innerprecond == "gapp":  
                    folder_path = os.path.join(
                        output_dir,
                        material,
                        f"nbands_{adjusted_nbands}",
                        f"spacing_{i}",
                        f"supercell_{'_'.join(map(str, supercell_factors))}",
                        f"precond_{precond}_{innerprecond}",
                        f"phase_{phase}",
                        f"gapp_pcg_{gapp_pcg_num}",
                        
                    )
                    os.makedirs(folder_path, exist_ok=True)

                elif precond == "shift-and-invert" and innerprecond == "Neumann":  
                    folder_path = os.path.join(
                        output_dir,
                        material,
                        f"nbands_{adjusted_nbands}",
                        f"spacing_{i}",
                        f"supercell_{'_'.join(map(str, supercell_factors))}",
                        f"precond_{precond}_{innerprecond}",
                        f"phase_{phase}",
                        f"Neumann_pcg_{Neumann_pcg_num}",
                        f"innerorder_{innerorder}",
                        
                    )
                    os.makedirs(folder_path, exist_ok=True)

                elif precond == "Neumann":
                    folder_path = os.path.join(
                        output_dir,
                        material,
                        f"nbands_{adjusted_nbands}",
                        f"spacing_{i}",
                        f"supercell_{'_'.join(map(str, supercell_factors))}",
                        f"precond_{precond}_{innerprecond}",
                        f"phase_{phase}",
                        f"outerorder_{outerorder}",
                        
                    )
                    os.makedirs(folder_path, exist_ok=True)
                else:
                    print("Not matched precond type")
                    exit(-1)

                for idx in range(3):

                    #calculation command (bypass None)
                    command = ["python", script]
                    add_opt(command, "--material", material)
                    add_opt(command, "--spacing", i)
                    add_opt(command, "--precond", precond)            
                    add_opt(command, "--nbands", int(nbands * np.prod(supercell_factors)))
                    add_opt(command, "--phase", phase)
                    # add_opt(command, "--density_filename", f"density_{material}.pt")
                    add_opt(command, "--outerorder", outerorder)
                    add_opt(command, "--inner", innerprecond)
                    add_opt(command, "--innerorder", innerorder)
                    add_opt(command, "--error_cutoff", error_cutoff)
                    add_opt(command, "--upf_files", upf_files)      
                    add_opt(command, "--pp_type", pp_type)
                    add_opt(command, "--dir", cif_file_path)
                    add_opt(command, "--supercell", list(supercell_factors))
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

                    stdout, stderr = process.communicate()
                    results.append(stdout)

                    print(f"Command executed: {command}")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")

                    # scf  iter 
                    match = re.search(r"Elapsed time\[davidson\]\s*:\s*([\d.]+)\s*sec", stdout)
                    if match:
                        diagonal_total_time = float(match.group(1))
                        diagonal_total_times.append(diagonal_total_time)

                    max_diagonal_iterations_list.append(
                            stdout.count("*=*=*=*=*=*=*=*=*=*=*=*=*=*i_iter")
                        )
                    #max_iterations_list.append(
                    #        stdout.count("cg iter=")
                    #        )
                    scf_count = stdout.count("==================== [ SCF CYCLE ")
                    max_scf_iteration_list.append(scf_count)

                    precond_matches = re.findall(
                             r"\(preconditioning\):\s*([\d.eE+-]+)\s*s", stdout
                    )
                    precond_times = list(map(float, precond_matches))
                    total_precond_times.append(sum(precond_times))


                if diagonal_total_times:
                    sorted_times = sorted(diagonal_total_times)

                # make labeled_results ordering total diagonalization time
                    if len(sorted_times) >= 3:
                        labeled_results = {
                            sorted_times[0]: "fast",
                            sorted_times[1]: "median",
                            sorted_times[2]: "slow",
                            }
                    elif len(sorted_times) == 2:
                        labeled_results = {
                            sorted_times[0]: "fast",
                            sorted_times[1]: "median",
                        }
                    elif len(sorted_times) == 1:
                        labeled_results = {sorted_times[0]: "fast"}
                    else:
                        labeled_results = {}

                    print(
                        f"Updated labeled_results: {labeled_results}"
                    )  # for debug

                    for idx, time in enumerate(diagonal_total_times):
                        if time in labeled_results:
                            label = labeled_results[time]
                        else:
                            closest_time = min(
                                labeled_results.keys(),
                                key=lambda k: abs(k - time),
                            )
                            label = labeled_results[closest_time]
                        
                        
                        result_filename = os.path.join(
                            folder_path,
                            f"{precond}_{material}_run{idx+1}_{label}_supercell_{'_'.join(map(str, supercell_factors))}.txt",
                        )

                        with open(result_filename, "w") as f:
                            f.write(results[idx])
                            f.write(
                                    f"\nSummary: material = {material}, spacing = {i}, nbands = {adjusted_nbands}, supercell = {supercell_factors}, preconditioner = {precond_type}, innerpreconditioner = {innerprecond}, outerorder = {outerorder}, innerorder = {innerorder}, error_cutoff = {error_cutoff}, gapp_pcg = {gapp_pcg_num}, Neumann_pcg = {Neumann_pcg_num}\n"
                                    )

                            

                    median_time = (
                        np.median(diagonal_total_times)
                        if diagonal_total_times
                        else "N/A"
                    )
                    max_iterations = (
                        max(max_diagonal_iterations_list) if max_diagonal_iterations_list else 0 
                    )

                    max_scf_iterations = (
                            np.median(max_scf_iteration_list) if max_scf_iteration_list else 0
                    )

                    total_precond_time = (
                        np.median(total_precond_times) if total_precond_times else 0
                    )

                    per_precond_time = (
                        total_precond_time / max_iterations
                        if max_iterations > 0
                        else "N/A"
                    )
                        

                with open(summary_file, "a") as f:
                    f.write(
                        f"material = {material}, spacing = {i}, nbands = {adjusted_nbands}, supercell = {supercell_factors}, preconditioner = {precond_type}, innerpreconditioner = {innerprecond}, outerorder = {outerorder}, innerorder = {innerorder}, error_cutoff = {error_cutoff}, gapp_pcg = {gapp_pcg_num}, Neumann_pcg = {Neumann_pcg_num}, total_diagonal_time = {median_time}, iteration = {max_iterations}, total_precond_time = {total_precond_time}, per_precond_time = {per_precond_time}, scf_iterations = {max_scf_iterations}\n")

print("all calculation saved")
