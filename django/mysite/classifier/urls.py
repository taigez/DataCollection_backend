from django.urls import path, re_path
from . import views

urlpatterns = [
    path('education/', views.show_edu, name='education'),
    path('interest/', views.show_int, name='interest'),
    path('awards/', views.show_awd, name='awards'),
    path('position/', views.show_pos, name='position'),
    path('new/', views.new, name='new'),
    path('reset_raw', views.reset_raw, name='reset_raw'),
    path('pending/', views.show_pending, name='pending'),
    path('mislabeled/', views.show_irrelevant, name='irrelevant'),
    path('performance/', views.show_performance, name='performance'),
    
    path('delete_edu/<int:id>', views.delete_edu, name='delete_edu'),
    path('delete_temp_edu/<int:id>', views.delete_temp_edu, name='delete_temp_edu'),
    path('delete_awd/<int:id>', views.delete_awd, name='delete_awd'),
    path('delete_temp_awd/<int:id>', views.delete_temp_awd, name='delete_temp_awd'),
    path('delete_int/<int:id>', views.delete_int, name='delete_int'),
    path('delete_temp_int/<int:id>', views.delete_temp_int, name='delete_temp_int'),
    path('delete_pos/<int:id>', views.delete_pos, name='delete_pos'),

    path('delete_irr_edu/<int:id>', views.delete_irr_edu, name='delete_irr_edu'),
    path('delete_irr_awd/<int:id>', views.delete_irr_awd, name='delete_irr_awd'),
    path('delete_irr_int/<int:id>', views.delete_irr_int, name='delete_irr_int'),

    path('save_temp_awd/<int:id>', views.save_temp_awd, name='save_temp_awd'),
    path('save_temp_edu/<int:id>', views.save_temp_edu, name='save_temp_edu'),
    path('save_temp_int/<int:id>', views.save_temp_int, name='save_temp_int'),
    path('movetemp_edu2awd/<int:id>', views.temp_edu2awd, name='tedu2awd'),
    path('movetemp_edu2int/<int:id>', views.temp_edu2int, name='tedu2int'),
    path('movetemp_int2awd/<int:id>', views.temp_int2awd, name='tint2awd'),
    path('movetemp_int2edu/<int:id>', views.temp_int2edu, name='tint2edu'),
    path('movetemp_awd2edu/<int:id>', views.temp_awd2edu, name='tawd2edu'),
    path('movetemp_awd2int/<int:id>', views.temp_awd2int, name='tawd2int'),
    path('move_edu2awd/<int:id>', views.edu2awd, name='edu2awd'),
    path('move_edu2int/<int:id>', views.edu2int, name='edu2int'),
    path('move_int2awd/<int:id>', views.int2awd, name='int2awd'),
    path('move_int2edu/<int:id>', views.int2edu, name='int2edu'),
    path('move_awd2edu/<int:id>', views.awd2edu, name='awd2edu'),
    path('move_awd2int/<int:id>', views.awd2int, name='awd2int'),
    re_path(r'^receive_json/$', views.handle_get, name='receive2'),
    path('delete_all_edu/', views.delete_all_edu, name='delete_all_edu'),
    path('delete_all_awd/', views.delete_all_awd, name='delete_all_awd'),
    path('delete_all_int/', views.delete_all_int, name='delete_all_int'),

    path('testing/', views.testing, name='testing'),
    path('reset', views.reset, name='reset'),
]