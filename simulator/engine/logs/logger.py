import os


class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path


    def log_loss(self, epoch, step, pg_loss, v_loss, entropy_loss, total_loss):
        log_entry = f"Epoch: {epoch}, Step: {step}, Policy Loss: {pg_loss:.4f}, Value Loss: {v_loss:.4f}, Entropy Loss: {entropy_loss:.4f}, Total Loss: {total_loss:.4f}\n"
        with open(self.log_file_path + 'ppo_loss.txt', 'a') as log_file:
            log_file.write(log_entry)

    def log_action(self, epoch, step, action):
        def count_occurrences(lst):
            count_dict = {}
            for item in lst:
                if item in count_dict:
                    count_dict[item] += 1
                else:
                    count_dict[item] = 1
            return count_dict

        counts = count_occurrences(action)
        log_entry = f"=======Epoch: {epoch}, Step: {step}=========\n"
        for value, count in counts.items():
            log_entry += f"{value}: {count} times\n"

        with open(self.log_file_path + 'action_records.txt', 'a') as log_file:
            log_file.write(log_entry)

    def clear_log(self):

        # List all files in the folder
        files = os.listdir(self.log_file_path)

        # Iterate through the files and delete .txt files
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(self.log_file_path, file)
                os.remove(file_path)
                print(f"Deleted {file_path}")

        print("All .txt files in the folder have been deleted.")
