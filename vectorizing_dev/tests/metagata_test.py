import requests
import json

# Пример входного чанка
chunk = {
    "document_id": "doc_001",
    "year": 1623,
    "doc_type": "chronicle",
    "text": (
"The dark-brown, strongly built sword-dancer in classical attire and a bright feathered crown, dancing (like King David) in honor of the Lord, is rarely seen even now. But when we consider all the other symbols of ceremonial splendor and priestly authority—the gold-embroidered banners, the heavy silver crosses and swinging censers, the rich garlands of flowers, and the palm branches—beneath the dark blue sky covering all, it must be admitted that the High Festivals of the Missions could rival in magnificence any Saint's day in Europe. One might almost imagine the stern spirit of Loyola's disciples still lingering about the quiet plaza and under the decaying verandas with their carved pillar capitals, which will never be restored once they finally succumb to the wear of wind and weather. Women in their long tipoyas glide silently by, balancing primitive ewers on their heads, and the men pass with a brief greeting. The convent-like stillness has not yet entirely subdued the children, who chatter, play, and ask unanswerable questions, as children do everywhere. Here, genuine indigenous people perform—partly on familiar, partly on strangely shaped instruments of their own making—the masterpieces of old Italian sacred music. With a diligence not often attributed to their race, they have preserved the art from generation to generation, despite the prolonged misrule of the white masters of the land, which would have crushed the artistic inclinations of a less resilient people. Who, after this, will deny them the capacity for further development? The noble features of this Indian, a member of one of our boat crews, always reminded me of Seume's 'A Canadian, who is still a European,' etc.; and if he did not quite match that ideal of a native, he was at least one of the most taciturn among the taciturn Indians. Mariano, a Mojos from Trinidad, was a handy, clever fellow who, under our cook's guidance, strove to expand his culinary skills and occasionally secure an especially good morsel. His broad cheekbones, slanted eyes, sparse beard, and tendency toward stoutness gave him the appearance of a Chinese mandarin, though somewhat darker in color. The old chief, who, together with his tribe, more than forty years ago left the forests on the Ivinheima and Iguatemy to play, more or less, the role of a mediatised prince in the Aldeamento de San Ignacio on the Parananapema, remains to this day a prototype of the good-natured, cunning Guarani. The dominance of the well-armed settlers, always ready for violence and drawing ever closer to his native woods, and perhaps vague memories and tales of the paternal government of the Jesuits, will have led him to the conviction that it is better to live under the protection of"
    )
}

# Вытаскиваем переменные
document_id = chunk["document_id"]
year = chunk["year"]
doc_type = chunk["doc_type"]
text = chunk["text"]

print(f"Document ID: {document_id}")
print(f"Year: {year}")
print(f"Type: {doc_type}")

# Делаем POST-запрос к локальному эндпоинту FastAPI
url = "http://192.168.168.5:8000/generate_metadata"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, json=chunk)

if response.ok:
    print("Ответ endpoint-а:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
else:
    print("Ошибка:", response.status_code, response.text)
