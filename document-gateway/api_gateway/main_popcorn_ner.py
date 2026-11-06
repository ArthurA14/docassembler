from document_common.components.base.popcorn_ner import PopcornNerClient

client = PopcornNerClient()
print(client.health())
print(client.ner("Egypt’s Ministry of Tourism said archaeologists discovered a tomb near Luxor on 27 September 2021." \
"Professor Salima Ikram from the American University in Cairo called it 'one of the most important finds since 1922'.", lang="eng").raw)

# {"text": "Egypt’s Ministry of Tourism said archaeologists discovered a tomb near Luxor on 27 September 2021." \
# "Professor Salima Ikram from the American University in Cairo called it 'one of the most important finds since 1922'.", "lang": "ENG"}
