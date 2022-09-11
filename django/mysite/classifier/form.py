from django.forms import ModelForm
from pathy import field
from .models import Sentences_awd, Sentences_edu, Sentences_int, Raw

class PostRawForm(ModelForm):
    class Meta:
        model = Raw
        fields = '__all__'

# class PostEduForm(ModelForm):
#     class Meta:
#         model = Sentences_edu
#         fields = '__all__'

# class PostIntForm(ModelForm):
#     class Meta:
#         model = Sentences_int
#         fields = '__all__'

# class PostAwdForm(ModelForm):
#     class Meta:
#         model = Sentences_awd
#         fields = '__all__'
