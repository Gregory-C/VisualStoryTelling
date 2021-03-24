import os
import io
import httplib2
from googleapiclient.http import MediaIoBaseDownload
from pprint import pprint
from oauth2client.file import Storage
from googleapiclient.discovery import build
from apiclient import errors
from apiclient import http as api_http
from oauth2client.client import OAuth2WebServerFlow

# make OUT_PATH where we will save the dataset
OUT_PATH = '/freespace/local/xc295'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# Copy your credentials from the console
# https://console.developers.google.com
CLIENT_ID = '1031915833238-qelsipmsj1besb2cfbpukcvmepa9i0l5.apps.googleusercontent.com'
CLIENT_SECRET = 'ELWoWLoKn5wdhHIdoRVL79hK'

OAUTH_SCOPE = 'https://www.googleapis.com/auth/drive'
REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'
CREDS_FILE = os.path.join(os.path.dirname(__file__), 'credentials.json')
storage = Storage(CREDS_FILE)
credentials = storage.get()

if credentials is None:
    # Run through the OAuth flow and retrieve credentials
    flow = OAuth2WebServerFlow(CLIENT_ID, CLIENT_SECRET, OAUTH_SCOPE, REDIRECT_URI)
    authorize_url = flow.step1_get_authorize_url()
    print('Go to the following link in your browser: ' + authorize_url)
    code = input('Enter verification code: ').strip()
    credentials = flow.step2_exchange(code)
    storage.put(credentials)

# Create an httplib2.Http object and authorize it with our credentials
http = httplib2.Http()
http = credentials.authorize(http)
service = build('drive', 'v2', http=http)




file_ids = [
    '0ByQS_kT8kViSZnZPY1dmaHJzMHc',
    '0ByQS_kT8kViSb0VjVDJ3am40VVE',
    '0ByQS_kT8kViSTmQtd1VfWWFyUHM',
    '0ByQS_kT8kViSQ1ozYmlITXlUaDQ',
    '0ByQS_kT8kViSTVY1MnFGV0JiVkk',
    '0ByQS_kT8kViSYmhmbnp6d2I4a2M',
    '0ByQS_kT8kViSZl9aNGVuX0llcEU',
    '0ByQS_kT8kViSWXJ3R3hsZllsNVk',
    '0ByQS_kT8kViSR2N4cFpweURhTjg',
    '0ByQS_kT8kViScllKWnlaVU53Skk',
    '0ByQS_kT8kViSV2QxZW1rVXcxT1U',
    '0ByQS_kT8kViSNGNPTEFhdGxkMnM',
    '0ByQS_kT8kViSWmtRa1lMcG1EaHc',
    '0ByQS_kT8kViSTHJ0cGxSVW1SRFk'
]

file_names = [
    'train_split.1.tar.gz',
    'train_split.2.tar.gz',
    'train_split.3.tar.gz',
    'train_split.4.tar.gz',
    'train_split.5.tar.gz',
    'train_split.6.tar.gz',
    'train_split.7.tar.gz',
    'train_split.8.tar.gz',
    'train_split.9.tar.gz',
    'train_split.10.tar.gz',
    'train_split.11.tar.gz',
    'train_split.12.tar.gz',
    'val_images.tar.gz',
    'test_images.tar.gz'
]

for file_id, file_name in zip(file_ids, file_names):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))

    fh.seek(0)
    with open(os.path.join(OUT_PATH, file_name), 'wb') as f:
        f.write(fh.read())
        f.close()


