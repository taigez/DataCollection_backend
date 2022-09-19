from cgitb import text
from functools import total_ordering
from sys import prefix
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect

# Create your views here.

import json
from django.contrib.auth.models import User #####
from django.http import JsonResponse , HttpResponse ####
from django.contrib import messages
from .models import Sentences_awd, Sentences_edu, Sentences_int, Sentences_temp_int, Sentences_temp_awd, Sentences_temp_edu, Sentences_irr_awd, Sentences_irr_edu, Sentences_irr_int
from .models import Predicted_total_awd, Predicted_total_edu, Predicted_total_int, Correct_total_awd, Correct_total_edu, Correct_total_int, True_total_awd, True_total_edu, True_total_int
from .models import Awd_data, Edu_data, Int_data
from .form import PostRawForm
from .classify import process_paragraph, train_awd, train_edu, train_int, get_csv
from django.views.decorators.csrf import csrf_exempt
import pandas as pd

def show_pending(request):
    total_pending = Sentences_temp_awd.objects.count() + Sentences_temp_edu.objects.count() + Sentences_temp_int.objects.count()
    return render(request, "pending.html", {"num_pending":total_pending,
                                            "temp_edu":Sentences_temp_edu.objects.all(),
                                            "temp_awd":Sentences_temp_awd.objects.all(),
                                            "temp_int":Sentences_temp_int.objects.all()})

def show_edu(request):
    return render(request, "edu.html", {"edu_sentences":Sentences_edu.objects.all()})

def show_int(request):
    return render(request, "interest.html", {"int_sentences":Sentences_int.objects.all()})

def show_awd(request):
    return render(request, "awards.html", {"awd_sentences":Sentences_awd.objects.all()})

def show_irrelevant(request):
    return render(request, "irrelevant.html", {"edu_sentences":Sentences_irr_edu.objects.all(),
                                        "int_sentences":Sentences_irr_int.objects.all(),
                                        "awd_sentences":Sentences_irr_awd.objects.all()})

def show_performance(request):
    backRec = get_object_or_404(True_total_edu, pk=1).body / get_object_or_404(Predicted_total_edu, pk=1).body
    intRec = get_object_or_404(True_total_int, pk=1).body / get_object_or_404(Predicted_total_int, pk=1).body
    awardRec = get_object_or_404(True_total_awd, pk=1).body / get_object_or_404(Predicted_total_awd, pk=1).body
    backPrec = get_object_or_404(Correct_total_edu, pk=1).body / get_object_or_404(Predicted_total_edu, pk=1).body
    intPrec = get_object_or_404(Correct_total_int, pk=1).body / get_object_or_404(Predicted_total_int, pk=1).body 
    awardPrec =  get_object_or_404(Correct_total_awd, pk=1).body / get_object_or_404(Predicted_total_awd, pk=1).body
    return render(request, "performance.html", {"backRec":backRec,
                                                "intRec":intRec,
                                                "awardRec":awardRec,
                                                "backPrec":backPrec,
                                                "intPrec":intPrec,
                                                "awardPrec":awardPrec})

def delete_edu(request, id):
    edu_sen = get_object_or_404(Sentences_edu, pk=id)
    edu_sen.delete()
    
    return redirect('/classifier/education')

def delete_int(request, id):
    int_sen = get_object_or_404(Sentences_int, pk=id)
    int_sen.delete()
    
    return redirect('/classifier/interest')

def delete_awd(request, id):
    awd_sen = get_object_or_404(Sentences_awd, pk=id)
    awd_sen.delete()
    
    return redirect('/classifier/awards')

def delete_irr_edu(request, id):
    edu_sen = get_object_or_404(Sentences_irr_edu, pk=id)
    edu_sen.delete()
    
    return redirect('/classifier/mislabeled')

def delete_irr_int(request, id):
    int_sen = get_object_or_404(Sentences_irr_int, pk=id)
    int_sen.delete()
    
    return redirect('/classifier/mislabeled')

def delete_irr_awd(request, id):
    awd_sen = get_object_or_404(Sentences_irr_awd, pk=id)
    awd_sen.delete()
    
    return redirect('/classifier/mislabeled')

def delete_temp_edu(request, id):
    pedu_sen = get_object_or_404(Sentences_temp_edu, pk=id)
    pedu_sen.delete()
    
    return redirect('/classifier/pending')

def delete_temp_int(request, id):
    pint_sen = get_object_or_404(Sentences_temp_int, pk=id)
    pint_sen.delete()
    
    return redirect('/classifier/pending')

def delete_temp_awd(request, id):
    pawd_sen = get_object_or_404(Sentences_temp_awd, pk=id)
    pawd_sen.delete()
    
    return redirect('/classifier/pending')

def delete_all_edu(request):
    Sentences_temp_edu.objects.all().delete()
    return redirect('/classifier/pending')

def delete_all_awd(request):
    Sentences_temp_awd.objects.all().delete()
    return redirect('/classifier/pending')

def delete_all_int(request):
    Sentences_temp_int.objects.all().delete()
    return redirect('/classifier/pending')

def save_temp_edu(request, id):
    pedu_sen = get_object_or_404(Sentences_temp_edu, pk=id)
    new_edu = Sentences_edu(body=pedu_sen.body)
    new_edu.save()
    pedu_sen.delete()   
    return redirect('/classifier/pending')

def save_temp_int(request, id):
    pint_sen = get_object_or_404(Sentences_temp_int, pk=id)
    new_int = Sentences_int(body=pint_sen.body)
    new_int.save()
    pint_sen.delete()
    
    return redirect('/classifier/pending')

def save_temp_awd(request, id):
    pawd_sen = get_object_or_404(Sentences_temp_awd, pk=id)
    new_awd = Sentences_awd(body=pawd_sen.body)
    new_awd.save()
    pawd_sen.delete()
    
    return redirect('/classifier/pending')

def temp_edu2awd(request, id):
    pedu_sen = get_object_or_404(Sentences_temp_edu, pk=id)
    new_pawd = Sentences_temp_awd(body = pedu_sen.body)
    new_pawd.save()
    pedu_sen.delete()
    return redirect('/classifier/pending')

def temp_edu2int(request, id):
    pedu_sen = get_object_or_404(Sentences_temp_edu, pk=id)
    new_pint = Sentences_temp_int(body = pedu_sen.body)
    new_pint.save()
    pedu_sen.delete()
    return redirect('/classifier/pending')

def temp_int2awd(request, id):
    pint_sen = get_object_or_404(Sentences_temp_int, pk=id)
    new_pawd = Sentences_temp_awd(body = pint_sen.body)
    new_pawd.save()
    pint_sen.delete()
    return redirect('/classifier/pending')

def temp_int2edu(request, id):
    pint_sen = get_object_or_404(Sentences_temp_int, pk=id)
    new_pedu = Sentences_temp_edu(body = pint_sen.body)
    new_pedu.save()
    pint_sen.delete()
    return redirect('/classifier/pending')

def temp_awd2edu(request, id):
    pawd_sen = get_object_or_404(Sentences_temp_awd, pk=id)
    new_pedu = Sentences_temp_edu(body = pawd_sen.body)
    new_pedu.save()
    pawd_sen.delete()
    return redirect('/classifier/pending')

def temp_awd2int(request, id):
    pawd_sen = get_object_or_404(Sentences_temp_awd, pk=id)
    new_pint = Sentences_temp_int(body = pawd_sen.body)
    new_pint.save()
    pawd_sen.delete()
    return redirect('/classifier/pending')

def edu2awd(request, id):
    edu_sen = get_object_or_404(Sentences_edu, pk=id)
    new_awd = Sentences_awd(body = edu_sen.body)
    new_awd.save()
    edu_sen.delete()
    return redirect('/classifier/education')

def edu2int(request, id):
    edu_sen = get_object_or_404(Sentences_edu, pk=id)
    new_int = Sentences_int(body = edu_sen.body)
    new_int.save()
    edu_sen.delete()
    return redirect('/classifier/education')

def int2awd(request, id):
    int_sen = get_object_or_404(Sentences_int, pk=id)
    new_awd = Sentences_awd(body = int_sen.body)
    new_awd.save()
    int_sen.delete()
    return redirect('/classifier/interest')

def int2edu(request, id):
    int_sen = get_object_or_404(Sentences_int, pk=id)
    new_edu = Sentences_edu(body = int_sen.body)
    new_edu.save()
    int_sen.delete()
    return redirect('/classifier/interest')

def awd2edu(request, id):
    awd_sen = get_object_or_404(Sentences_awd, pk=id)
    new_edu = Sentences_edu(body = awd_sen.body)
    new_edu.save()
    awd_sen.delete()
    return redirect('/classifier/awards')

def awd2int(request, id):
    awd_sen = get_object_or_404(Sentences_awd, pk=id)
    new_int = Sentences_int(body = awd_sen.body)
    new_int.save()
    awd_sen.delete()
    return redirect('/classifier/awards')

def new(request):
    if request.method == 'POST':
        raw_form = PostRawForm(request.POST)
        if raw_form.is_valid():
            obj = raw_form.save(commit=False)
            input_parapgraph = obj.body
            output_dict = process_paragraph(input_parapgraph)
            for edu_sen in output_dict["background"]:
                new_edu = Sentences_temp_edu(body=edu_sen)
                new_edu.save()

            for awd_sen in output_dict["awards"]:
                new_awd = Sentences_temp_awd(body=awd_sen)
                new_awd.save()

            for int_sen in output_dict["interest"]:
                new_int = Sentences_temp_int(body=int_sen)
                new_int.save()
            return redirect('/classifier/new/')
    else:
        raw_form = PostRawForm()
    return render(request, "new.html", {"raw_form":raw_form})


def check_training():
    if Sentences_awd.objects.count() >= 50:
        train_awd([sen.body for sen in Sentences_awd.objects.all()])
        Sentences_awd.objects.all().delete()

    if Sentences_edu.objects.count() >= 50:
        train_edu([sen.body for sen in Sentences_edu.objects.all()])
        Sentences_edu.objects.all().delete()

    if Sentences_int.objects.count() >= 50:
        train_int([sen.body for sen in Sentences_int.objects.all()])
        Sentences_awd.objects.all().delete()

def process_temp_edu(all_tempEdu, wrong_sentences):
    for temp in all_tempEdu:
        vaild = True
        for wrong in wrong_sentences:
            # something is wrong with current prediction temp
            if temp == wrong[0]:
                vaild = False
                # save temp to the irrelevant model
                new_irr_edu = Sentences_irr_edu(body=temp)
                new_irr_edu.save()
                # save temp under the correct model
                if wrong[1] == "awards":
                    new_awd = Sentences_awd(body=wrong[0])
                    new_awd.save()
                elif wrong[1] == "interest":
                    new_int = Sentences_int(body=wrong[0])
                    new_int.save()
                elif wrong[1] == "none":
                    new_irr_awd = Sentences_irr_awd(body=temp)
                    new_irr_awd.save()
                    new_irr_int = Sentences_irr_int(body=temp)
                    new_irr_int.save()
                
                wrong_sentences.remove(wrong)

        # nothing wrong with this prediction, save it
        if vaild:
            new_edu = Sentences_edu(body=temp)
            new_edu.save()
    
    Sentences_temp_edu.objects.all().delete()
    return wrong_sentences

def process_temp_awd(all_tempAwd, wrong_sentences):
    for temp in all_tempAwd:
        vaild = True
        for wrong in wrong_sentences:
            # something is wrong with current prediction temp
            if temp == wrong[0]:
                vaild = False
                # save temp to the irrelevant model
                new_irr_awd = Sentences_irr_awd(body=temp)
                new_irr_awd.save()
                # save temp under the correct model
                if wrong[1] == "background":
                    new_edu = Sentences_edu(body=wrong[0])
                    new_edu.save()
                elif wrong[1] == "interest":
                    new_int = Sentences_int(body=wrong[0])
                    new_int.save()
                elif wrong[1] == "irrelevant":
                    new_irr_edu = Sentences_irr_edu(body=temp)
                    new_irr_edu.save()
                    new_irr_int = Sentences_irr_int(body=temp)
                    new_irr_int.save()
                
                wrong_sentences.remove(wrong)

        # nothing wrong with this prediction, save it
        if vaild:
            new_awd = Sentences_awd(body=temp)
            new_awd.save()
    
    Sentences_temp_awd.objects.all().delete()
    return wrong_sentences

def process_temp_int(all_tempInt, wrong_sentences):
    for temp in all_tempInt:
        vaild = True
        for wrong in wrong_sentences:
            # something is wrong with current prediction temp
            if temp == wrong[0]:
                vaild = False
                # save temp to the irrelevant model
                new_irr_int = Sentences_irr_int(body=temp)
                new_irr_int.save()
                # save temp under the correct model
                if wrong[1] == "background":
                    new_edu = Sentences_edu(body=wrong[0])
                    new_edu.save()
                elif wrong[1] == "awards":
                    new_awd = Sentences_awd(body=wrong[0])
                    new_awd.save()
                elif wrong[1] == "irrelevant":
                    new_irr_edu = Sentences_irr_edu(body=temp)
                    new_irr_edu.save()
                    new_irr_awd = Sentences_irr_awd(body=temp)
                    new_irr_awd.save()
                
                wrong_sentences.remove(wrong)

        # nothing wrong with this prediction, save it
        if vaild:
            new_int = Sentences_int(body=temp)
            new_int.save()
    
    Sentences_temp_int.objects.all().delete()
    return wrong_sentences

def process_numbers(predicted_total_awd, predicted_total_edu, predicted_total_int, correct_total_awd, correct_total_edu, correct_total_int, true_total_awd, true_total_edu, true_total_int):
    pta = get_object_or_404(Predicted_total_awd, pk=1)
    pta.body += int(predicted_total_awd)
    pta.save()

    pte = get_object_or_404(Predicted_total_edu, pk=1)
    pte.body += int(predicted_total_edu)
    pte.save()

    pti = get_object_or_404(Predicted_total_int, pk=1)
    pti.body += int(predicted_total_int)
    pti.save()

    cta = get_object_or_404(Correct_total_awd, pk=1)
    cta.body += int(correct_total_awd)
    cta.save()

    cte = get_object_or_404(Correct_total_edu, pk=1)
    cte.body += int(correct_total_edu)
    cte.save()

    cti = get_object_or_404(Correct_total_int, pk=1)
    cti.body += int(correct_total_int)
    cti.save()

    tta = get_object_or_404(True_total_awd, pk=1)
    tta.body += int(true_total_awd)
    tta.save()

    tte = get_object_or_404(True_total_edu, pk=1)
    tte.body += int(true_total_edu)
    tte.save()

    tti = get_object_or_404(True_total_int, pk=1)
    tti.body += int(true_total_int)
    tti.save()
    

@csrf_exempt
def handle_get(request):
    input_parapgraph = request.GET.get('text', None)
    input_labels = request.GET.get('data', None)
    predicted_total_awd = request.GET.get('Predicted_total_awd', None)
    predicted_total_edu = request.GET.get('Predicted_total_edu', None)
    predicted_total_int = request.GET.get('Predicted_total_int', None)
    correct_total_awd = request.GET.get('Correct_total_awd', None)
    correct_total_edu = request.GET.get('Correct_total_edu', None)
    correct_total_int = request.GET.get('Correct_total_int', None)
    true_total_awd = request.GET.get('True_total_awd', None)
    true_total_edu = request.GET.get('True_total_edu', None)
    true_total_int = request.GET.get('True_total_int', None)
    
    
    if (input_labels == None):
        # the first button, parse
        output_dict, total_sentneces = process_paragraph(input_parapgraph)

        # save parsed sentences into temp storage
        for edu_sen in output_dict['background']:
            new_edu = Sentences_temp_edu(body=edu_sen)
            new_edu.save()
        
        for awd_sen in output_dict['awards']:
            new_awd = Sentences_temp_awd(body=awd_sen)
            new_awd.save()

        for int_sen in output_dict['interest']:
            new_int = Sentences_temp_int(body=int_sen)
            new_int.save()

        data = {
            'background' : output_dict['background'],
            'awards' : output_dict['awards'],
            'interest' : output_dict['interest'],
            'bnum' : len(output_dict['background']),
            'anum' : len(output_dict['awards']),
            'inum' : len(output_dict['interest']),
            'none' : total_sentneces - len(output_dict['background']) - len(output_dict['awards']) - len(output_dict['interest']),
        }

        return JsonResponse(data)
    else:
        # the second button, only wrongly highlighted sentences will appear here
        input_labels = json.loads(input_labels)

        all_tempEdu = [sen.body for sen in Sentences_temp_edu.objects.all()]
        all_tempAwd = [sen.body for sen in Sentences_temp_awd.objects.all()]
        all_tempInt = [sen.body for sen in Sentences_temp_int.objects.all()]
        wrong_sentences = [sen for sen in input_labels]

        wrong_sentences = process_temp_edu(all_tempEdu, wrong_sentences)
        wrong_sentences = process_temp_awd(all_tempAwd, wrong_sentences)
        wrong_sentences = process_temp_int(all_tempInt, wrong_sentences)
            
        # TO-DO 
        # if wrong_sentences is not empty, which means that there might be new sentences or temp is mismatched
        process_numbers(predicted_total_awd, predicted_total_edu, predicted_total_int, correct_total_awd, correct_total_edu, correct_total_int, true_total_awd, true_total_edu, true_total_int)
        
        data = {
            'backRec' : get_object_or_404(True_total_edu, pk=1).body / get_object_or_404(Predicted_total_edu, pk=1).body,
            'intRec' : get_object_or_404(True_total_int, pk=1).body / get_object_or_404(Predicted_total_int, pk=1).body,
            'awardRec' : get_object_or_404(True_total_awd, pk=1).body / get_object_or_404(Predicted_total_awd, pk=1).body,
            'backPrec' : get_object_or_404(Correct_total_edu, pk=1).body / get_object_or_404(Predicted_total_edu, pk=1).body,
            'intPrec' : get_object_or_404(Correct_total_int, pk=1).body / get_object_or_404(Predicted_total_int, pk=1).body,
            'awardPrec' :  get_object_or_404(Correct_total_awd, pk=1).body / get_object_or_404(Predicted_total_awd, pk=1).body,
        }

        #check_training()

        return JsonResponse(data)

def testing(request):
    

    return HttpResponse("OK!")
