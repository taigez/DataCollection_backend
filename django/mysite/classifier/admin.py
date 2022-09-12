from django.contrib import admin

# Register your models here.

from .models import Raw, Sentences_awd, Sentences_edu, Sentences_int, Sentences_temp_int, Sentences_temp_awd, Sentences_temp_edu, Sentences_irr_awd, Sentences_irr_edu, Sentences_irr_int
from .models import Predicted_total_awd, Predicted_total_edu, Predicted_total_int, Correct_total_awd, Correct_total_edu, Correct_total_int, True_total_awd, True_total_edu, True_total_int

admin.site.register(Raw)
admin.site.register(Sentences_int)
admin.site.register(Sentences_awd)
admin.site.register(Sentences_edu)
admin.site.register(Sentences_temp_int)
admin.site.register(Sentences_temp_awd)
admin.site.register(Sentences_temp_edu)
admin.site.register(Sentences_irr_int)
admin.site.register(Sentences_irr_edu)
admin.site.register(Sentences_irr_awd)
admin.site.register(Predicted_total_int)
admin.site.register(Predicted_total_awd)
admin.site.register(Predicted_total_edu)
admin.site.register(Correct_total_awd)
admin.site.register(Correct_total_edu)
admin.site.register(Correct_total_int)
admin.site.register(True_total_awd)
admin.site.register(True_total_edu)
admin.site.register(True_total_int)
