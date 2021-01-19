def printProgressBar (iteration, total, prefix, decimals = 1, length = 25, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix}: |{bar}| {percent}% Complete', end = "\r")
    # Print New Line on Complete
    if iteration == total: 
        print()