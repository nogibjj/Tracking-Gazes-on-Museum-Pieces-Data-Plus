def repository_details(paths_in_txt_files):
    """This function extracts the paths from the Paths.txt file

    It returns the folder path with all the participant folders
    and the name of the art piece"""
    with open("Paths.txt", "r") as f:
        for line in f.readlines():
            if "AFTER THE COLON" in line and "PARTICIPANT" in line:
                participant_repository = line.split(":", maxsplit=1)[1].strip()

                print(f"participant_repository is {participant_repository}")

            elif "AFTER THE COLON" in line and "ART PIECE" in line:
                name_of_art_piece = line.split(":", maxsplit=1)[1].strip()

        return participant_repository, name_of_art_piece
