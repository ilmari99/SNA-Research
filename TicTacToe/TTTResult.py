from RLFramework import Result
import numpy as np


class TTTResult(Result):
    
    def save_array_to_file(self, arr: np.ndarray, file_path: str) -> None:
        with open(file_path, "a") as f:
            # Save all values as int, except the last one, which is a float.
            fmt = ["%d" for _ in range(arr.shape[1] - 1)] + ["%f"]
            np.savetxt(f, arr, delimiter=",", fmt=fmt)