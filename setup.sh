#!/bin/bash

mkdir -p weights/AGFusion

FILE_ID="1-RIl45od71fM0rWieW8YhOAfrqWeAE9q"
DEST_PATH="weights/AGFusion/AGFusionModel_best.pth"

if [ ! -f "$DEST_PATH" ]; then
    echo "Скачивание весов модели..."
    wget --quiet --show-progress --no-check-certificate \
    "https://drive.google.com/uc?id=$FILE_ID&export=download" -O "$DEST_PATH"
    
    # Проверка успешности загрузки
    if [ $? -eq 0 ]; then
        echo "Веса успешно загружены в $DEST_PATH"
    else
        echo "Ошибка загрузки! Проверьте ID файла и соединение."
        rm -f "$DEST_PATH"  # Удаляем частично загруженный файл
    fi
else
    echo "Файл весов уже существует: $DEST_PATH"
fi
