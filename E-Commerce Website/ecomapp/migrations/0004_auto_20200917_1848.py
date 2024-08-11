# Generated by Django 3.1 on 2020-09-17 13:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ecomapp', '0003_auto_20200915_2051'),
    ]

    operations = [
        migrations.AlterField(
            model_name='order',
            name='payment_method',
            field=models.CharField(choices=[('Cash On Delivery', 'Cash On Delivery'), ('Khalti', 'Khalti'), ('Esewa', 'Esewa')], default='Cash On Delivery', max_length=20),
        ),
    ]
