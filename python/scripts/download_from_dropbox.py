import argparse

import dropbox
import os


class downlod_from_dropbox():
    '''
    Dropbox downloader - Fetches a file from Dropbox and saves into specified  directory.
    Currently suuports loading files.

    Usage:
    Create Dropbox App in Dropbox website:
    1. Go to: https://www.dropbox.com/developers/apps
    2. Create App
    3. Click generate at: Generated access token
    4. Init class with tocken: dropbox_downloader = download_from_dropbox(<Generated_access_token>)
    5. Fetch and save file:    dropbox_downloader.download_zip_file(dropbox_data_file_path, save_path)

    Good reference:
    https://github.com/dropbox/dropbox-sdk-python/blob/master/example/updown.py

    TODO: Add support for loading a directory
    '''

    def __init__(self, dropbox_tocken):
        self.dropbox_tocken = dropbox_tocken
        self.dbx = dropbox.Dropbox(dropbox_tocken)
        self.dbx.users_get_current_account()

    def print_dropbox_files(self):
        for entry in self.dbx.files_list_folder('').entries:
            print(entry.name)

    def download_zip_file(self, dropbox_data_file_path, save_path):
        if os.path.exists(save_path):
            print('Dataset already exists. Skipping download')
            pass
        md, res = self.dbx.files_download(dropbox_data_file_path)
        # res.content
        print('Found file in Dropbox account. Attemp to download')
        with open(save_path, "wb") as f:
            junk = b"\xCC" * 1028
            f.write(res.content)

        print('Finished downloading file. Saving to ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropbox-tocken',
                        default="2jtIURYzJrAAAAAAAAAIcmGv_DvWlhGjtrXkEFhMv5pvcs8GzL3XP3U5qbNZYw-4",
                        help="Amount of vertical splits")
    parser.add_argument('--dropbox-data-file-path',
                        default="/Datasets/Gaze/GeraNet/GeraNet_18-08-01_18-26-31.zip",
                        help="Download data from Dropbox. Check out the Dropbox account to locate the data")
    parser.add_argument('--save-path',
                        default="gera_data.zip",
                        help="Save data path in computer")
    args = parser.parse_args()

    # Usage:
    dropbox_downloader = downlod_from_dropbox(dropbox_tocken=args.dropbox_tocken)
    # dropbox_downloader.download_zip_file(dropbox_data_file_path='/Datasets/Gaze/GazeCapture/00120.zip', save_path='00120.zip')
    # dropbox_downloader.download_zip_file(dropbox_data_file_path='/Datasets/Gaze/GazeCapture/GazeCapture.tar', save_path='~/datasets/GazeCapture.tar')
    dropbox_downloader.download_zip_file(dropbox_data_file_path=args.dropbox_data_file_path, save_path=args.save_path)
