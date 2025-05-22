import os

from core_mixtral import CoreMixtral

if __name__ == "__main__":
    model = CoreMixtral()
    prefill_time, decode_time = model.generate(
        "University of Washington is", output_token=20
    )
    print(
        f"prefill_time: {prefill_time}, decode_time: {decode_time}"
    )