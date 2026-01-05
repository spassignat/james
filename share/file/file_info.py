
class FileInfo:
    path: str
    relative_path: str
    filename: str
    extension: str
    directory: str
    size: int

    def __init__(self, path: str, relative_path: str, filename: str, extension: str, directory: str, size: int):
        super().__init__()
        self.path = path
        self.relative_path = relative_path
        self.filename = filename
        self.extension = extension
        self.directory = directory
        self.size = size
