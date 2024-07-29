import re
import os
import argparse
import matplotlib.pyplot as plt

def parse_file(file_path):
    try:
        with open(file_path, 'r',encoding="utf-8") as file:
            content = file.read()

        # Extract the last val_mae value
        val_mae_matches = re.findall(r'- val_mae: (\d+\.\d+)', content)
        if val_mae_matches:
            last_val_mae = float(val_mae_matches[-4])
        else:
            last_val_mae = None
            
        val_loss_matches = re.findall(r'- val_loss: (\d+\.\d+)', content)
        if val_loss_matches:
            last_val_loss = float(val_loss_matches[-4])
        else:
            last_val_loss = None

        # Extract the average score for PentobiNNPlayer
        avg_score_match = re.search(r"Average score.*'PlayerToTest': (\d+\.\d+)", content)
        if not avg_score_match:
            avg_score_match = re.search(r"Average score.*'PentobiNNPlayer': (\d+\.\d+)", content)
        if avg_score_match:
            avg_score = float(avg_score_match.group(1))
        else:
            avg_score = None

        # Extract the win rate for PentobiNNPlayer
        win_rate_match = win_rate_match = re.search(r"Wins.*'PlayerToTest': (\d+)", content)
        if not win_rate_match:
            win_rate_match = re.search(r"Wins.*'PentobiNNPlayer': (\d+)", content)
        if win_rate_match:
            win_rate = float(win_rate_match.group(1))
        else:
            win_rate = None
        
        if (last_val_loss is None) and (last_val_mae is None):
            print(f"Error parsing file {file_path}: last_val_loss or last_val_mae is None: {last_val_loss=} {last_val_mae=}")
        
        if any((val is None for val in [avg_score, win_rate])):
            print(f"Error parsing file {file_path}: val_mae={last_val_mae}, avg_score={avg_score}, win_rate={win_rate}")
            return None
        
        num_params_match = re.search(r"Total params: (\d+)", content)
        if num_params_match:
            num_params = int(num_params_match.group(1))
        else:
            num_params = None

        return {
            'file': file_path,
            'val_mae': last_val_mae,
            'val_loss' : last_val_loss,
            'avg_score': avg_score,
            'win_rate': win_rate,
            'num_params': num_params,
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Given a directory, parse all *.out files
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing the *.out files")
    parser.add_argument("--x_axis", help="X-axis value to plot", default="val_mae")
    parser.add_argument("--y_axis", help="Y-axis value to plot", default="avg_score")
    parser.add_argument("--str_filter", help="Regular expression to filter the files", default=".out")
    parser.add_argument("--output", help="Output file to save the results", default="val_mae_vs_win_rate.png")
    args = parser.parse_args()
    directory = os.path.abspath(args.directory)
    print(f"Directory: ", directory)

    files = [(root, subdirs, files) for root, subdirs, files in os.walk(directory)]
    results = []
    for root, subdirs, files in files:
        for file in files:
            if args.str_filter not in file:
                continue
            print(f"Processing file: {file}")
            file = os.path.join(root, file)
            result = parse_file(file)
            if result:
                results.append(result)
                #print(f"Result: {result['file']}: {result['avg_score']}"
    results = sorted(results, key=lambda x: x[args.x_axis] if x[args.x_axis] is not None else 0)
    # Print the results
    for result in results:
        print(result)

    # Extract val_mae and win_rate for the scatter plot
    x_values = [result[args.x_axis] for result in results]
    y_values = [result[args.y_axis] for result in results]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.7)
    plt.xlabel(args.x_axis)
    #plt.xlim(400000, 600000)
    plt.ylabel(args.y_axis)
    plt.title('Scatter plot of ' + args.x_axis + ' vs ' + args.y_axis)
    plt.grid(True)
    plt.savefig(args.output)
    print('Scatter plot saved as ', args.output)
