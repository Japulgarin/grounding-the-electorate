"""Plain-language guide copy. Recorded persona text is never translated."""
COPY = {
    'en': {
        'titles':['A master’s thesis, explained in five steps','The experiment','Three synthetic voters','Who wins the United States?','El Salvador and Brazil','Explore the atlas'],
        'bodies':[
            'Can an AI simulate an electorate? Pick a language to follow the experiment from one synthetic voter to a whole country. Saved model answers stay in the language they were generated in.',
            'A master’s thesis on whether an AI can simulate an electorate. The model gpt-oss-20b receives one synthetic persona at a time, built from NVIDIA Nemotron personas grounded in real census data, and answers an election question with a probability for each candidate and a reason. This is not a poll.',
            'Each synthetic persona answers the same question twice. The first prompt contains only its demographic data; the second one adds more context about who the person is. Here you can see whether its vote stays the same or changes.',
            'Adding up 200,023 synthetic voters, state by state, gives a simulated Electoral College. Each context produces a different winner and margin. The real 2024 result is the benchmark.',
            'The same experiment ran in El Salvador 2024 (50,000 personas, Spanish and English) and Brazil 2026 (50,000 personas, Portuguese and English). El Salvador has a real result to compare with; Brazil is still pending.',
            'Start with the election map, then follow individual voters, check the extra experiments, or read about the thesis. This guide can be reopened from the header.'],
        'pillars':['Model','Personas','Elections','Contexts'],'pillar_notes':['gpt-oss-20b, temperature 0','NVIDIA Nemotron, census-grounded','USA 2024 · El Salvador 2024 · Brazil 2026','Demographics → + cultural background → + full persona'],
        'flow':['Synthetic persona','gpt-oss-20b','Vote + reason'],'flow_notes':['age, state, education…','one prompt, one answer','probability per candidate'],
        'cases':['Kept the vote','{a} → {b}','{b} → {a}'],
        'start':'Next','next':'Next','open':'Open the atlas','prompt_changed':'What changes in the prompt','adds':'adds','national':'National result','back':'Back','skip':'Go to the app','language':'Your language','reopen':'Quick guide','guide':'Quick guide',
        'synthetic':'Synthetic persona','previous_person':'Previous persona','next_person':'Next persona',
        'before':'Demographics only','cultural':'+ Cultural background','persona':'+ Full persona','career':'+ Career and Big Five','richer':'Choose the extra context',
        'reason':'Model-generated reason','details':'See the added profile text','age':'Age','sex':'Gender','education':'Education','occupation':'Occupation',
        'sample':'Saved model outputs; no new model calls. Not a representative sample.',
        'original':'Profile values and explanations remain in the original experiment language.',
        'changed':'Top choice changed','same':'Top choice stayed the same','tie':'Tie','probabilities':'Model probabilities',
        'context':'Context','winner':'Simulated winner','observed':'Real result','no_result':'No real result yet (election in October 2026)','simulated':'Simulated','states_matched':'states matched',
        'ev_note':'Electoral votes: all of each state’s electors go to its simulated winner, Maine and Nebraska included (as in the thesis notebooks). Popular vote uses population-weighted state means. Real shares are normalized to the simulated candidates.',
        'election':'Election view','election_desc':'Pick a country and a context, see the winner on the map, click a territory and open individual synthetic voters.',
        'changes':'See the changes','changes_desc':'Follow the same voter from one context to the next, or compare whole maps and vote flows.',
        'experiments':'Extra experiments','experiments_desc':'Does the vote change with the prompt language? Does the model vote for the name or for the programme?',
        'about':'About the thesis','about_desc':'Research questions, hypotheses, findings, and limits.',
        'ten':'10 personas','charts':'Maps and results','advanced':'Comparison settings','missing':'No matched saved examples are available for this comparison.',
        'questions':'What is the thesis asking?',
        'hypotheses':['Can a model reproduce national and regional election patterns?','Does a richer persona improve alignment, or just change the answer?','How much do language and prompt design affect the result?'],
        'findings':'More detail is not automatically better. Cultural context improves the documented USA comparison, while fuller persona text can move results farther from the observed outcome. A convincing explanation is not proof of a realistic voter.',
        'deep':'Read the hypotheses and evidence','methods':'Methods and sources',
    },
    'es': {
        'titles':['Una tesis de maestría, explicada en cinco pasos','El experimento','Tres votantes sintéticos','¿Quién gana en Estados Unidos?','El Salvador y Brasil','Explora el atlas'],
        'bodies':[
            '¿Puede una IA simular un electorado? Elige un idioma para seguir el experimento desde un votante sintético hasta un país entero. Las respuestas guardadas del modelo se quedan en el idioma en que se generaron.',
            'Una tesis de maestría sobre si una IA puede simular un electorado. El modelo gpt-oss-20b recibe una persona sintética a la vez, construida con personas de NVIDIA Nemotron basadas en datos reales del censo, y responde una pregunta electoral con una probabilidad por candidato y una razón. No es una encuesta.',
            'Cada persona sintética responde la misma pregunta dos veces. El primer prompt solo contiene sus datos demográficos; el segundo añade más contexto sobre quién es. Aquí puedes ver si su voto se mantiene o cambia.',
            'Sumando 200.023 votantes sintéticos, estado por estado, sale un Colegio Electoral simulado. Cada contexto produce un ganador y un margen distintos. El resultado real de 2024 es la referencia.',
            'El mismo experimento se hizo en El Salvador 2024 (50.000 personas, español e inglés) y en Brasil 2026 (50.000 personas, portugués e inglés). El Salvador tiene un resultado real con el que comparar; Brasil todavía no.',
            'Empieza por el mapa electoral y luego sigue votantes individuales, revisa los experimentos adicionales o lee sobre la tesis. Esta guía se puede reabrir desde la cabecera.'],
        'pillars':['Modelo','Personas','Elecciones','Contextos'],'pillar_notes':['gpt-oss-20b, temperatura 0','NVIDIA Nemotron, basadas en el censo','EE. UU. 2024 · El Salvador 2024 · Brasil 2026','Demografía → + contexto cultural → + persona completa'],
        'flow':['Persona sintética','gpt-oss-20b','Voto + razón'],'flow_notes':['edad, estado, educación…','un prompt, una respuesta','probabilidad por candidato'],
        'cases':['Mantuvo el voto','{a} → {b}','{b} → {a}'],
        'start':'Siguiente','next':'Siguiente','open':'Abrir el atlas','prompt_changed':'Qué cambia en el prompt','adds':'añade','national':'Resultado nacional','back':'Atrás','skip':'Ir a la app','language':'Tu idioma','reopen':'Guía rápida','guide':'Guía rápida',
        'synthetic':'Persona sintética','previous_person':'Persona anterior','next_person':'Siguiente persona',
        'before':'Solo demografía','cultural':'+ Contexto cultural','persona':'+ Persona completa','career':'+ Carrera y Big Five','richer':'Elige el contexto adicional',
        'reason':'Razón generada por el modelo','details':'Ver el texto añadido al perfil','age':'Edad','sex':'Género','education':'Educación','occupation':'Ocupación',
        'sample':'Respuestas guardadas del modelo; no se hacen llamadas nuevas. No es una muestra representativa.',
        'original':'Los valores del perfil y las explicaciones conservan el idioma del experimento.',
        'changed':'Cambió la primera opción','same':'Se mantuvo la primera opción','tie':'Empate','probabilities':'Probabilidades del modelo',
        'context':'Contexto','winner':'Ganador simulado','observed':'Resultado real','no_result':'Aún no hay resultado real (elección en octubre de 2026)','simulated':'Simulado','states_matched':'estados acertados',
        'ev_note':'Votos electorales: todos los votos de cada estado van a su ganador simulado, incluidos Maine y Nebraska (como en los notebooks de la tesis). El voto popular usa medias estatales ponderadas por población. Las cuotas reales se normalizan a los candidatos simulados.',
        'election':'Vista electoral','election_desc':'Elige país y contexto, mira el ganador en el mapa, pulsa un territorio y abre votantes sintéticos individuales.',
        'changes':'Ver los cambios','changes_desc':'Sigue al mismo votante de un contexto al siguiente, o compara mapas completos y flujos de voto.',
        'experiments':'Experimentos adicionales','experiments_desc':'¿Cambia el voto con el idioma del prompt? ¿El modelo vota por el nombre o por el programa?',
        'about':'Sobre la tesis','about_desc':'Preguntas de investigación, hipótesis, hallazgos y límites.',
        'ten':'10 personas','charts':'Mapas y resultados','advanced':'Ajustes de comparación','missing':'No hay ejemplos guardados y emparejados para esta comparación.',
        'questions':'¿Qué pregunta la tesis?',
        'hypotheses':['¿Puede el modelo reproducir patrones electorales nacionales y regionales?','¿Una persona más completa mejora el ajuste o solo cambia la respuesta?','¿Cuánto afectan el idioma y el diseño del prompt al resultado?'],
        'findings':'Más detalle no siempre es mejor. El contexto cultural mejora la comparación documentada de EE. UU., pero una persona más completa puede alejar los resultados de la elección observada. Una explicación convincente no demuestra un votante realista.',
        'deep':'Leer las hipótesis y la evidencia','methods':'Métodos y fuentes',
    },
    'pt': {
        'titles':['Uma dissertação de mestrado, explicada em cinco passos','O experimento','Três eleitores sintéticos','Quem vence nos Estados Unidos?','El Salvador e Brasil','Explore o atlas'],
        'bodies':[
            'Uma IA pode simular um eleitorado? Escolha um idioma para acompanhar o experimento de um eleitor sintético até um país inteiro. As respostas salvas do modelo ficam no idioma em que foram geradas.',
            'Uma dissertação de mestrado sobre se uma IA pode simular um eleitorado. O modelo gpt-oss-20b recebe uma persona sintética por vez, construída com personas NVIDIA Nemotron baseadas em dados reais do censo, e responde a uma pergunta eleitoral com uma probabilidade por candidato e uma razão. Não é uma pesquisa de opinião.',
            'Cada persona sintética responde à mesma pergunta duas vezes. O primeiro prompt contém apenas seus dados demográficos; o segundo acrescenta mais contexto sobre quem ela é. Aqui você pode ver se o voto se mantém ou muda.',
            'Somando 200.023 eleitores sintéticos, estado por estado, surge um Colégio Eleitoral simulado. Cada contexto produz um vencedor e uma margem diferentes. O resultado real de 2024 é a referência.',
            'O mesmo experimento foi feito em El Salvador 2024 (50.000 personas, espanhol e inglês) e no Brasil 2026 (50.000 personas, português e inglês). El Salvador tem um resultado real para comparar; o Brasil ainda não.',
            'Comece pelo mapa eleitoral e depois acompanhe eleitores individuais, veja os experimentos adicionais ou leia sobre a tese. Este guia pode ser reaberto pelo cabeçalho.'],
        'pillars':['Modelo','Personas','Eleições','Contextos'],'pillar_notes':['gpt-oss-20b, temperatura 0','NVIDIA Nemotron, baseadas no censo','EUA 2024 · El Salvador 2024 · Brasil 2026','Demografia → + contexto cultural → + persona completa'],
        'flow':['Persona sintética','gpt-oss-20b','Voto + razão'],'flow_notes':['idade, estado, escolaridade…','um prompt, uma resposta','probabilidade por candidato'],
        'cases':['Manteve o voto','{a} → {b}','{b} → {a}'],
        'start':'Próximo','next':'Próximo','open':'Abrir o atlas','prompt_changed':'O que muda no prompt','adds':'acrescenta','national':'Resultado nacional','back':'Voltar','skip':'Ir para o app','language':'Seu idioma','reopen':'Guia rápido','guide':'Guia rápido',
        'synthetic':'Persona sintética','previous_person':'Persona anterior','next_person':'Próxima persona',
        'before':'Só demografia','cultural':'+ Contexto cultural','persona':'+ Persona completa','career':'+ Carreira e Big Five','richer':'Escolha o contexto adicional',
        'reason':'Razão gerada pelo modelo','details':'Ver o texto adicionado ao perfil','age':'Idade','sex':'Gênero','education':'Escolaridade','occupation':'Ocupação',
        'sample':'Respostas salvas do modelo; nenhuma nova chamada. Não é uma amostra representativa.',
        'original':'Valores do perfil e explicações permanecem no idioma original do experimento.',
        'changed':'A primeira escolha mudou','same':'A primeira escolha não mudou','tie':'Empate','probabilities':'Probabilidades do modelo',
        'context':'Contexto','winner':'Vencedor simulado','observed':'Resultado real','no_result':'Ainda não há resultado real (eleição em outubro de 2026)','simulated':'Simulado','states_matched':'estados acertados',
        'ev_note':'Votos eleitorais: todos os votos de cada estado vão para o vencedor simulado, incluindo Maine e Nebraska (como nos notebooks da tese). O voto popular usa médias estaduais ponderadas pela população. As parcelas reais são normalizadas aos candidatos simulados.',
        'election':'Visão eleitoral','election_desc':'Escolha país e contexto, veja o vencedor no mapa, clique num território e abra eleitores sintéticos individuais.',
        'changes':'Ver as mudanças','changes_desc':'Acompanhe o mesmo eleitor de um contexto ao seguinte, ou compare mapas inteiros e fluxos de voto.',
        'experiments':'Experimentos adicionais','experiments_desc':'O voto muda com o idioma do prompt? O modelo vota no nome ou no programa?',
        'about':'Sobre a tese','about_desc':'Perguntas de pesquisa, hipóteses, resultados e limites.',
        'ten':'10 personas','charts':'Mapas e resultados','advanced':'Ajustes de comparação','missing':'Nenhum exemplo salvo e pareado para esta comparação.',
        'questions':'O que a tese pergunta?',
        'hypotheses':['O modelo reproduz padrões eleitorais nacionais e regionais?','Uma persona mais completa melhora o alinhamento ou só muda a resposta?','Quanto o idioma e o desenho do prompt afetam o resultado?'],
        'findings':'Mais detalhes não são sempre melhores. O contexto cultural melhora a comparação documentada dos EUA, mas uma persona mais completa pode afastar os resultados da eleição observada. Uma explicação convincente não prova um eleitor realista.',
        'deep':'Ler as hipóteses e evidências','methods':'Métodos e fontes',
    },
    'de': {
        'titles':['Eine Masterarbeit, erklärt in fünf Schritten','Das Experiment','Drei synthetische Wähler','Wer gewinnt die USA?','El Salvador und Brasilien','Den Atlas erkunden'],
        'bodies':[
            'Kann eine KI eine Wählerschaft simulieren? Wählen Sie eine Sprache, um dem Experiment von einer synthetischen Person bis zu einem ganzen Land zu folgen. Gespeicherte Modellantworten bleiben in ihrer Ursprungssprache.',
            'Eine Masterarbeit zur Frage, ob eine KI eine Wählerschaft simulieren kann. Das Modell gpt-oss-20b erhält jeweils eine synthetische Person, erstellt aus NVIDIA-Nemotron-Personen auf Basis echter Zensusdaten, und beantwortet eine Wahlfrage mit einer Wahrscheinlichkeit je Kandidat und einer Begründung. Das ist keine Umfrage.',
            'Jede synthetische Person beantwortet dieselbe Frage zweimal. Der erste Prompt enthält nur ihre demografischen Daten; der zweite ergänzt mehr Kontext zu ihrer Person. Hier sehen Sie, ob ihre Stimme gleich bleibt oder sich ändert.',
            'Aus 200.023 synthetischen Wählern, Bundesstaat für Bundesstaat, entsteht ein simuliertes Electoral College. Jeder Kontext liefert einen anderen Sieger und Abstand. Das echte Ergebnis 2024 ist der Maßstab.',
            'Dasselbe Experiment lief in El Salvador 2024 (50.000 Personen, Spanisch und Englisch) und Brasilien 2026 (50.000 Personen, Portugiesisch und Englisch). Für El Salvador gibt es ein echtes Ergebnis zum Vergleich; Brasilien steht noch aus.',
            'Beginnen Sie mit der Wahlkarte, folgen Sie dann einzelnen Wählern, prüfen Sie die Zusatzexperimente oder lesen Sie über die Arbeit. Diese Einführung lässt sich über die Kopfzeile erneut öffnen.'],
        'pillars':['Modell','Personen','Wahlen','Kontexte'],'pillar_notes':['gpt-oss-20b, Temperatur 0','NVIDIA Nemotron, zensusbasiert','USA 2024 · El Salvador 2024 · Brasilien 2026','Demografie → + kultureller Hintergrund → + volles Profil'],
        'flow':['Synthetische Person','gpt-oss-20b','Stimme + Begründung'],'flow_notes':['Alter, Bundesstaat, Bildung…','ein Prompt, eine Antwort','Wahrscheinlichkeit je Kandidat'],
        'cases':['Stimme blieb','{a} → {b}','{b} → {a}'],
        'start':'Weiter','next':'Weiter','open':'Atlas öffnen','prompt_changed':'Was sich im Prompt ändert','adds':'ergänzt','national':'Nationales Ergebnis','back':'Zurück','skip':'Zur App','language':'Ihre Sprache','reopen':'Kurze Einführung','guide':'Kurze Einführung',
        'synthetic':'Synthetische Person','previous_person':'Vorherige Person','next_person':'Nächste Person',
        'before':'Nur Demografie','cultural':'+ Kultureller Hintergrund','persona':'+ Volles Profil','career':'+ Karriere und Big Five','richer':'Zusätzlichen Kontext wählen',
        'reason':'Modellgenerierte Begründung','details':'Ergänzten Profiltext ansehen','age':'Alter','sex':'Geschlecht','education':'Bildung','occupation':'Beruf',
        'sample':'Gespeicherte Modellantworten ohne neue Aufrufe. Keine repräsentative Stichprobe.',
        'original':'Profilwerte und Begründungen bleiben in der ursprünglichen Versuchssprache.',
        'changed':'Erstpräferenz geändert','same':'Erstpräferenz unverändert','tie':'Gleichstand','probabilities':'Modellwahrscheinlichkeiten',
        'context':'Kontext','winner':'Simulierter Sieger','observed':'Echtes Ergebnis','no_result':'Noch kein echtes Ergebnis (Wahl im Oktober 2026)','simulated':'Simuliert','states_matched':'Staaten richtig',
        'ev_note':'Wahlleute: Alle Stimmen eines Staates gehen an seinen simulierten Sieger, auch in Maine und Nebraska (wie in den Notebooks der Arbeit). Die Wählerstimmen nutzen bevölkerungsgewichtete Staatenmittel. Echte Anteile sind auf die simulierten Kandidaten normiert.',
        'election':'Wahlansicht','election_desc':'Land und Kontext wählen, Sieger auf der Karte sehen, Region anklicken und einzelne synthetische Wähler öffnen.',
        'changes':'Veränderungen sehen','changes_desc':'Demselben Wähler von einem Kontext zum nächsten folgen oder ganze Karten und Stimmflüsse vergleichen.',
        'experiments':'Zusatzexperimente','experiments_desc':'Ändert sich die Stimme mit der Prompt-Sprache? Wählt das Modell den Namen oder das Programm?',
        'about':'Über die Masterarbeit','about_desc':'Forschungsfragen, Hypothesen, Befunde und Grenzen.',
        'ten':'10 Personen','charts':'Karten und Ergebnisse','advanced':'Vergleichseinstellungen','missing':'Keine passenden gespeicherten Beispiele für diesen Vergleich.',
        'questions':'Was untersucht die Masterarbeit?',
        'hypotheses':['Kann das Modell nationale und regionale Wahlmuster reproduzieren?','Verbessert ein ausführlicheres Profil die Übereinstimmung oder verändert es nur die Antwort?','Wie stark wirken Sprache und Prompt-Gestaltung auf das Ergebnis?'],
        'findings':'Mehr Details sind nicht automatisch besser. Kultureller Kontext verbessert den dokumentierten USA-Vergleich; ausführlichere Profile können Ergebnisse aber von der beobachteten Wahl entfernen. Eine überzeugende Begründung beweist keinen realistischen Wähler.',
        'deep':'Hypothesen und Belege lesen','methods':'Methoden und Quellen',
    },
}


# Stitch redesign (2026-09-18): step 1 leads with the question; the old title becomes the eyebrow
# (six screens, not five); the "reopen" sentence of step 6 moves into its own note.
_SIX={'en':('five steps','six steps'),'es':('cinco pasos','seis pasos'),'pt':('cinco passos','seis passos'),'de':('fünf Schritten','sechs Schritten')}
_EXTRA={
    'en':dict(step_of='Step {n} of {t}',stat_elections='elections analysed',stat_personas='synthetic personas',stat_model='base model',fact_sheet='Fact sheet',thesis_tag='Master’s thesis'),
    'es':dict(step_of='Paso {n} de {t}',stat_elections='elecciones analizadas',stat_personas='personas sintéticas',stat_model='modelo base',fact_sheet='Ficha técnica',thesis_tag='Tesis de maestría'),
    'pt':dict(step_of='Passo {n} de {t}',stat_elections='eleições analisadas',stat_personas='personas sintéticas',stat_model='modelo base',fact_sheet='Ficha técnica',thesis_tag='Dissertação de mestrado'),
    'de':dict(step_of='Schritt {n} von {t}',stat_elections='analysierte Wahlen',stat_personas='synthetische Personen',stat_model='Basismodell',fact_sheet='Steckbrief',thesis_tag='Masterarbeit'),
}
for _lang,_c in COPY.items():
    _old,_new=_SIX[_lang]
    _c['eyebrow']=_c['titles'][0].replace(_old,_new)
    _q,_rest=_c['bodies'][0].split('? ',1)
    _c['titles']=[_q+'?']+_c['titles'][1:]
    _c['bodies']=[_rest]+_c['bodies'][1:]
    _body,_note=_c['bodies'][5].rsplit('. ',1)
    _c['bodies'][5]=_body+'.'
    _c['reopen_note']=_note
    _c.update(_EXTRA[_lang])

# Findings, rewritten after the corrected runs (19 Sep 2026): persona text creates diverse voters.
COPY['en']['findings'] = ('Richer personas create a far more diverse electorate. With demographics alone the model gives almost everyone the same few answers; '
                          'with the cultural background or the full persona, votes spread across candidates, groups and regions, and every explanation fits its voter. '
                          'That diversity does not reproduce the real result, but it makes persona prompts a useful tool for other experiments that need varied, '
                          'plausible voters, such as testing how different kinds of people react to a message, an issue or a programme.')
COPY['es']['findings'] = ('Las personas más ricas crean un electorado mucho más diverso. Solo con datos demográficos el modelo da casi a todos las mismas pocas respuestas; '
                          'con el trasfondo cultural o la persona completa, los votos se reparten entre candidatos, grupos y regiones, y cada explicación encaja con su votante. '
                          'Esa diversidad no reproduce el resultado real, pero convierte los prompts con persona en una herramienta útil para otros experimentos que necesiten '
                          'votantes variados y verosímiles, por ejemplo probar cómo reaccionan distintos tipos de personas a un mensaje, un tema o un programa.')
COPY['pt']['findings'] = ('Personas mais ricas criam um eleitorado muito mais diverso. Só com dados demográficos o modelo dá quase a todos as mesmas poucas respostas; '
                          'com o contexto cultural ou a persona completa, os votos se distribuem entre candidatos, grupos e regiões, e cada explicação combina com seu eleitor. '
                          'Essa diversidade não reproduz o resultado real, mas torna os prompts com persona uma ferramenta útil para outros experimentos que precisem de '
                          'eleitores variados e plausíveis, por exemplo testar como diferentes tipos de pessoas reagem a uma mensagem, um tema ou um programa.')
COPY['de']['findings'] = ('Reichere Personen erzeugen eine viel vielfältigere Wählerschaft. Nur mit demografischen Angaben gibt das Modell fast allen dieselben wenigen Antworten; '
                          'mit kulturellem Hintergrund oder vollständiger Persona verteilen sich die Stimmen über Kandidaten, Gruppen und Regionen, und jede Begründung passt zu ihrer Person. '
                          'Diese Vielfalt bildet das reale Ergebnis nicht nach, macht Persona-Prompts aber zu einem nützlichen Werkzeug für andere Experimente, die vielfältige, '
                          'plausible Wähler brauchen, etwa um zu testen, wie verschiedene Menschen auf eine Botschaft, ein Thema oder ein Programm reagieren.')
