import taskplan

def create_app():
    return taskplan.run([
        taskplan.Project(".", "WatteNNTask")
], 1)