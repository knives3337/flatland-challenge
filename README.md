## flatland-challenge
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

> Nustatoma dabartinio skripto katalogo vieta
script_dir = os.path.dirname(os.path.abspath(__file__))

> Nurodomas kelias iki treniravimo duomenų rinkinio failo
data_path = os.path.join(script_dir, 'flatland_train.npz')

> Užkraunami treniravimo duomenys
data = np.load(data_path)  
X = data['X']  # Paveikslėlių duomenys
y = data['y']  # Klasės (etiketės) duomenys

> Normalizuojami, pagerinami paveikslėlių duomenys (pikselių reikšmės skalėje nuo 0 iki 1)
X = X / 255.0  

> Koreguojamos etiketės reikšmės: jei reikšmė nėra 0, ji sumažinama 2
y[y != 0] -= 2  

> Sukuriamas konvoliucinis neuroninis tinklas su sluoksniais
model = Sequential([
    Input(shape=(50, 50, 1)),  # Įvesties sluoksnis su paveikslėlių dydžiu (50x50 pikseliai, 1 kanalas - pilkas)
    Conv2D(8, (3, 3), activation='relu'),  # Konvoliucinis sluoksnis su 8 filtrai ir aktyvavimo funkcija 'relu'
    MaxPooling2D(pool_size=(2, 2)),  # Maksimalaus sumažinimo sluoksnis (sumažina išvesties dydį perpus)
    Conv2D(16, (3, 3), activation='relu'),  # Antras konvoliucinis sluoksnis su 16 filtrų
    MaxPooling2D(pool_size=(2, 2)),  # Antras maksimalus sumažinimas
    Flatten(),  # Ištiesina duomenis, kad jie būtų naudojami tankiuose sluoksniuose
    Dense(16, activation='relu'),  # Tankusis (fully-connected) sluoksnis su 16 neuronų ir 'relu' aktyvacija
    Dense(5, activation='softmax')  # Išvesties sluoksnis su 5 klasėmis ir 'softmax' aktyvacija
])

 > Modelio kompiliavimas su optimizatoriumi ir nuostolių funkcija
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

 > Modelio treniravimas su duomenimis (10 epochų, po 32 pavyzdžius per paketą)
model.fit(X, y, epochs=10, batch_size=32)

 > Išsaugo modelį į failą. Išsaugo modelį h5 formatu. Tai leidžia išsaugoti modelio svorius, sluoksnius, parametrus ir treniravimo konfigūraciją, kad vėliau būtų galima modelį įkelti ir naudoti be papildomo treniravimo.
model_path = os.path.join(script_dir, 'compact_model.h5')
model.save(model_path)

 > Apskaičiuoja ir išspausdina modelio failo dydį KB
model_size = os.path.getsize(model_path) / 1024  
print(f'Model size: {model_size:.2f} KB')

