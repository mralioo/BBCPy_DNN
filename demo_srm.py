from datasets.preprocess_utils import generate_session_metadata

srm_data_root_path = "./local/data/SMR/raw"

if __name__ == "__main__":

    generate_session_metadata(srm_data_root_path)