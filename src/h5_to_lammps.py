import h5py
import numpy as np
import argparse

def read_box_edges(f_in):
    return f_in["/particles/fluid/box/edges"][()]

def wrap_pbc(positions, box_edges):
    """
    Wraps coordinates into the centered box [-L/2, L/2].
    """
    # Extract diagonal L from box edges
    L = np.diag(box_edges)

    return ((positions + L/2) % L) - L/2

def read_positions(f_in, time_step, box):
    dataset = f_in["/particles/fluid/position/value"]

    if time_step >= dataset.shape[0]:
        print(f"Warning: Time Step {time_step} is out of bounds.")
        return None

    pos_fluid = dataset[time_step]

    pos_fluid_wrapped = wrap_pbc(pos_fluid, box)
    species_fluid = ["F"] * pos_fluid.shape[0]

    n_atoms = pos_fluid.shape[0]

    particle_dict = {"positions" : pos_fluid_wrapped,
                     "num_atoms" : n_atoms,
                     "species" : species_fluid
                     }
    return particle_dict

def write_positions_lammps(f_out, particle_dict, time_step, box_edges):
    # Extract box lengths
    L = np.diag(box_edges)
    Lx, Ly, Lz = L[0], L[1], L[2]

    xlo, xhi = -Lx/2, Lx/2
    ylo, yhi = -Ly/2, Ly/2
    zlo, zhi = -Lz/2, Lz/2

    # Standard LAMMPS Dump Header
    f_out.write("ITEM: TIMESTEP\n")
    f_out.write(f"{time_step}\n")
    f_out.write("ITEM: NUMBER OF ATOMS\n")
    f_out.write(f"{particle_dict['num_atoms']}\n")
    f_out.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    f_out.write(f"{xlo:.6f} {xhi:.6f} 0.0\n")
    f_out.write(f"{ylo:.6f} {yhi:.6f} 0.0\n")
    f_out.write(f"{zlo:.6f} {zhi:.6f} 0.0\n")
    f_out.write("ITEM: ATOMS id type x y z\n")

    for i, pos in enumerate(particle_dict["positions"]):
        f_out.write(f"{i+1} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

def traj_count(f_in):
    return f_in["/particles/fluid/position/value"].shape[0]


def main():

    parser = argparse.ArgumentParser(description="Read positions from H5\
            simulation file and write in XYZ format")
    parser.add_argument("--input", required=True, help="H5 input path to read positions from")
    parser.add_argument("--output", required=True, help="XYZ output path")
    parser.add_argument("--range", type=int, nargs=2, default=[0, -1], help="Range of simulation snapshots to read")
    parser.add_argument("--every", type=int, default=1, help="Read every x steps")
    args = parser.parse_args()

    with h5py.File(args.input, "r") as f_in, open(args.output, "w") as f_out:
        total_frames = traj_count(f_in)

        start_frame, end_frame = args.range
        if end_frame == -1:
            end_frame = total_frames

        box = read_box_edges(f_in)

        print(f"Converting frames {start_frame} to {end_frame} every {args.every} steps...")

        count = 0
        for t in range(start_frame, end_frame, args.every):
            data = read_positions(f_in, time_step=t, box = box)
            if data is not None:
                write_positions_lammps(f_out, data, t, box)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} frames...", end='\r')

    print(f"\nDone! Extracted {count} frames to {args.output}")

if __name__ == "__main__":
    main()



