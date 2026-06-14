# Bienvenue 

## Peut-on prédire une coupure de courant avant qu'elle se produise ?

C'est la question que j'ai posée dans mon dernier projet data science — appliqué au réseau électrique togolais.

J'ai construit un pipeline complet :
- Prévision de charge (SARIMA + LSTM)
- Classification des blackouts (7 modèles ML)
- Analyse de risque par région et par heure
- Simulation de scénarios opérationnels

Meilleur modèle : Random Forest → F1 = 0.811 | AUC = 0.99
Découverte clé : la température du transformateur est le signal d'alarme #1

Le réseau togolais mérite des outils intelligents. Ce projet est ma contribution.
