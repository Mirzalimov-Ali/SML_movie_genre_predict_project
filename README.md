# 🎬 Predicting Movie Genre

## 🧾 Projectning asosiy maqsadi

- Bu project movie malumotlarini tahlil qilib, ularning **genre** larini aniqlashga qaratilgan.  
- (`multi-class classification`) 

* **Type:** Multi-class classification  
* **Target:** `Genre`  

---

## 📥 Data Extraction (Web scraping)

- **Source:** IMDb Lists:  
  - Horror: [HORROR_LIST_URL](https://www.imdb.com/list/ls021161997/)  
  - Action: [ACTION_LIST_URL](https://www.imdb.com/list/ls070233852/)  
  - Family: [FAMILY_LIST_URL](https://www.imdb.com/es-es/list/ls096388064/)  
  - Sport: [SPORT_LIST_URL](https://www.imdb.com/list/ls054945956/)  

- **What was done:** Ushbu saytlardan film nomi, tavsifi va boshqa atributlarni **web scraping** orqali oldim va DataFrame sifatida saqladim.  

* Har bir filmga ketma-ket `id` berildi.  
* Har bir film uchun **genre** atributi aniqlab olindi.  
* DataFrame keyingi **feature engineering** bosqichiga tayyorlandi.  

---

## 🔧 Feature Engineering — Overview

| Feature name          | Type        | Description                                                                 |
|----------------------|------------|-----------------------------------------------------------------------------|
| Title                 | text        | Film nomi                                                                   |
| Description           | text        | Film qisqacha tavsifi                                                       |
| Director              | categorical | Film rejissyori                                                             |
| MainCast              | categorical | Asosiy aktyorlar                                                           |
| Runtime               | numerical   | Film davomiyligi (daqiqa)                                                  |
| ReleaseYear           | numerical   | Chiqarilgan yil                                                            |
| Votes                 | numerical   | IMDb foydalanuvchi ovozlari                                                |
| Rating                | numerical   | IMDb reytingi                                                              |
| Genre                 | categorical | Maqsadli ustun — film turkumi (Horror, Action, Family, Sport)             |

---

## ⚙️ Preprocessing (Auto Pipeline)

`Preprocessing` classi datasetni **model-ready** qilish uchun yozilgan:  

- Automatic detection: **numerical** va **categorical** features  
- Steps: fill missing values → encode categorical → scale numeric → log-transform skewed features  
- Output: **ready-to-model dataset**  
- Fully **pipeline-based** va **robust** (try/except + logging)

---

## 🧠 Model Selection — Overview

- **LogisticRegression** (`sklearn.linear_model`)  
- **DecisionTreeClassifier** (`sklearn.tree`)  
- **RandomForestClassifier** (`sklearn.ensemble`)  
- **GradientBoostingClassifier** (`sklearn.ensemble`)  
- **XGBClassifier** (`xgboost`)  
- **KNeighborsClassifier** (`sklearn.neighbors`)  
- **ExtraTreesClassifier** (`sklearn.ensemble`)  
- **AdaBoostClassifier** (`sklearn.ensemble`)  

---

