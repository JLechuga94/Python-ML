from pydub import AudioSegment
import os

glob_div_counter = 0

patient_type = "abnormal"
database = "MITDATASET/"
# specific_database_name = "Atraining_{}/".format(patient_type)
specific_database_name = "training-a/{}/".format(patient_type)

global_path = "../../../../../../Downloads/datasets/"
folder_path = global_path + database + specific_database_name
new_path = global_path + database + "training-a/" + "divided_{}/".format(patient_type)

files_names = sorted(os.listdir(folder_path))[1:]
n_sec_segments = 3

print("DATABASE SELECTED FOR DIVISION", database)
for file in files_names:
    t1 = 0 * 1000
    t2 = n_sec_segments * 1000
    local_div_counter = 0
    print("\nFile processed: " + file)
    file_id = file.split(".")[0]
    original_audio = AudioSegment.from_wav(folder_path + file)
    length = original_audio.duration_seconds
    n_slices = int(length/n_sec_segments)
    print("Original audio length in seconds ", length)
    for i in range(n_slices + 1):
        newAudio = original_audio[t1:t2]
        if newAudio.duration_seconds == 3:
            newAudio.export(new_path + "{}_{}{}".format(file_id, i, ".wav"), format="wav")
            local_div_counter += 1
            glob_div_counter += 1
        t1 = t2
        t2 = t2 + (n_sec_segments * 1000)
    print("Divided files created ", local_div_counter)

print("\n# of files as input processing ", len(files_names))
print("# of files created from division ", glob_div_counter)
