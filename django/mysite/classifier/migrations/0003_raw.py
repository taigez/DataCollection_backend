# Generated by Django 4.0.6 on 2022-07-28 23:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classifier', '0002_sentences_edu_sentences_int_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Raw',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('body', models.CharField(max_length=5000)),
            ],
        ),
    ]
