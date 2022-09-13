# Generated by Django 4.0.6 on 2022-09-12 22:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0005_count_awd_count_edu_count_int_precision_awd_and_more'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Recall_int',
            new_name='Correct_total_awd',
        ),
        migrations.RenameModel(
            old_name='Recall_edu',
            new_name='Correct_total_edu',
        ),
        migrations.RenameModel(
            old_name='Recall_awd',
            new_name='Correct_total_int',
        ),
        migrations.RenameModel(
            old_name='Count_awd',
            new_name='Predicted_total_awd',
        ),
        migrations.RenameModel(
            old_name='Precision_awd',
            new_name='Predicted_total_edu',
        ),
        migrations.RenameModel(
            old_name='Precision_int',
            new_name='Predicted_total_int',
        ),
        migrations.RenameModel(
            old_name='Count_edu',
            new_name='True_total_awd',
        ),
        migrations.RenameModel(
            old_name='Precision_edu',
            new_name='True_total_edu',
        ),
        migrations.RenameModel(
            old_name='Count_int',
            new_name='True_total_int',
        ),
    ]
