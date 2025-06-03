# to run the virtual environment
# for mac: source venv/bin/activate
# for windows: venv\Scripts\activate

# to set environment variable (once per)
# mac: export FLASK_APP=web_app.py
# windows: set FLASK_APP=web_app.py

# to run website on personal server
# just run the script

# import statements
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, send_file
from werkzeug.utils import secure_filename
import secrets
import os
from datetime import timedelta
import threading

import shutil
import tempfile

import pandas as pd
from process_data import *
from file_delete import MonitorTime

# handling file type so we only take csv
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# set app and secret key
app = Flask(__name__)
app.config['SECRET_KEY'] = '703ad2f69c2b223b2409b38ecd54d80b' #16 bit random chars

# handling upload size
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit

# set session limit to 30 min
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# allow files to be sent from user_image directory
@app.route('/user_images/<filename>')
def user_images(filename):
    return send_from_directory('user_images', filename, as_attachment=True)

# deleting files at the proper time
def Start_File_Cleanup_Task():
    thread = threading.Thread(target=MonitorTime, daemon=True)
    thread.start()

# set routes
@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def main_page():
    if request.method == 'POST':
        # extract all of the parameters from the form - HERE
        inherent_std = float(request.form.get('param1'))
        std_guess = float(request.form.get('param2'))
        apodization_const = float(request.form.get('param3'))
        smoothing_points = float(request.form.get('param4'))
        deriv_points = float(request.form.get('param5'))
        upper_bound = float(request.form.get('param9'))
        lower_bound = float(request.form.get('param8'))
        member_title = request.form.get('param6')
        spectrum_title = request.form.get('param7')
        
        # get the file as object
        file = request.files['file']
        # make sure it exists and is an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # make new file name
            uniq_ext = secrets.token_urlsafe(16) # random string
            new_filename = f'{filename.split('.')[0]}_{uniq_ext}.csv'
            # download file
            file.save(os.path.join('user_files', new_filename))
            # load in file to pandas 
            try:
                df = pd.read_csv(os.path.join('user_files', new_filename), header=None)
                # process file
                structure, condensed_structure, processed_graph, whole_graph, process_df, curve_df, content_df = ProteinPipeline(df, sigma=inherent_std, std_guess=std_guess, L=apodization_const, points_smooth=smoothing_points, points_derive=deriv_points, title=member_title, spectrum_title=spectrum_title, upper_bound=upper_bound, lower_bound=lower_bound)
                # convert data into lists
                structure_names, structure_contents, cond_structure_names, cond_structure_contents = ProcessStructures(structure, condensed_structure)
                # prepare and save csvs
                # make users directory in the user csvs
                data_files_name = f'processed_files_{uniq_ext}'
                csv_path = os.path.join('user_csvs', data_files_name)
                os.mkdir(csv_path)
                # save each of the dfs to it
                process_df.to_csv(os.path.join(csv_path, "Amide_I_profiles.csv"))
                curve_df.to_csv(os.path.join(csv_path, "Curve_parameters.csv"), index=False)
                content_df.to_csv(os.path.join(csv_path, "Secondary_structure_content.csv"), index=False)
                # send information to and render the home page
                process_ok = True
                return render_template('home.html', processed=process_ok, structure_names=structure_names, structure_contents=structure_contents,
                                       cond_structure_names=cond_structure_names, cond_structure_contents=cond_structure_contents,
                                       processed_graph=processed_graph, whole_graph=whole_graph, data_files_name=data_files_name)
            except: # if file doesn't load properly
                failed_processing = True
                # load page accordingly
                return render_template('home.html', file_not_loaded = failed_processing)
        else:
            file_not_saved = False
            # load page accordingly
            return render_template('home.html', file_not_saved = file_not_saved)
        
    failed_process = False
    return render_template('home.html')

@app.route("/tutorial")
def tutorial():
    return render_template('tutorial.html')

@app.route("/about")
def about():
    return render_template('about.html')

# downloading compressed csvs code
@app.route('/download_zip/<path:foldername>', methods=['GET'])
def download_zip(foldername):
    # Path to the folder you want to zip
    folder_path = os.path.join('user_csvs', foldername)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        abort(404)  # Return 404 if the folder doesn't exist

    # Create a temporary zip file
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        # Zip the folder into the temporary file
        shutil.make_archive(os.path.splitext(temp_zip.name)[0], 'zip', folder_path)
        
        # Send the temporary zip file to the user for download - will self-delete after download
        return send_file(temp_zip.name, as_attachment=True, download_name=f'{foldername}.zip')

# run the html live for testing
if __name__=='__main__':
    Start_File_Cleanup_Task()
    app.run(port=8000, debug=True)