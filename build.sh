#!/usr/bin/env bash
# build.sh

# فعال‌سازی محیط مجازی اختیاری
# source venv/bin/activate

# مهاجرت دیتابیس
python manage.py makemigrations
python manage.py migrate

# جمع‌آوری فایل‌های استاتیک
python manage.py collectstatic --noinput
