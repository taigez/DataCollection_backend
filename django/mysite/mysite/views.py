from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import authenticate, login
from django.contrib import messages
from classifier.models import Sentences_awd, Sentences_edu, Sentences_int, Sentences_temp_awd, Sentences_temp_edu, Sentences_temp_int, Sentences_pos

# Create your views here.
def welcome(request):
    total_pending = Sentences_temp_awd.objects.count() + Sentences_temp_edu.objects.count() + Sentences_temp_int.objects.count()
    return render(request, "website/welcome.html",
                    {"num_pending":total_pending,
                    "num_edu_sentences":Sentences_edu.objects.count(),
                    "num_int_sentences":Sentences_int.objects.count(),
                    "num_awd_sentences":Sentences_awd.objects.count(),
                    "num_pos_sentences":Sentences_pos.objects.count()})