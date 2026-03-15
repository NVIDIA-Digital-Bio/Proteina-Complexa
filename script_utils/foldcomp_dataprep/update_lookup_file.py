import argparse
import os


def process_files(id_file_path, db_file_path):
    outdated_file_path = db_file_path + "_outdated"
    os.rename(db_file_path, outdated_file_path)

    with open(id_file_path) as id_file:
        id_set = set(line.strip() for line in id_file)

    with (
        open(outdated_file_path) as db_file,
        open(db_file_path, "w") as output_file,
    ):
        for line in db_file:
            fields = line.strip().split()
            if len(fields) >= 2 and fields[1] in id_set:
                output_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process db.lookup and id_db.txt files.")
    parser.add_argument("id_db", help="Path to the id_db.txt file")
    parser.add_argument("db_lookup", help="Path to the db.lookup file")
    args = parser.parse_args()

    process_files(args.id_db, args.db_lookup)
