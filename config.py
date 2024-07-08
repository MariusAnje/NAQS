ARCH_SPACE = {
    "filter_height": (1, 3, 5, 7),
    "filter_width": (1, 3, 5, 7),
    'stride_height': (1, 2, 3),
    'stride_width': (1, 2, 3),
    "num_filters": (24, 36, 48, 64),
    "pool_size": (1, 2)
    }

QUAN_SPACE = {
    "act_num_int_bits": (0, 1, 2, 3),
    "act_num_frac_bits": (0, 1, 2, 3, 4, 5, 6),
    "weight_num_int_bits": (0, 1, 2, 3,),
    "weight_num_frac_bits": (0, 1, 2, 3, 4, 5, 6)
    }

CLOCK_FREQUENCY = 100e6

N_NODES = 10

CIRCUIT_SPACE = {
    "choices": tuple(range(N_NODES * 2 + 2)),
    }




if __name__ == '__main__':
    print("architecture space: ", ARCH_SPACE)
    print("quantization space: ", QUAN_SPACE)
