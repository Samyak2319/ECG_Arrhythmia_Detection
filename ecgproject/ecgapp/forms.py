from django import forms

class ECGUploadForm(forms.Form):
    ecg_file = forms.FileField(label="Upload ECG CSV")

