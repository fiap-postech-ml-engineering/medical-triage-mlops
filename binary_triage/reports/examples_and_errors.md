# Exemplos de previsão e análise de erros (threshold=0.40)

## Exemplos corretos

- **Real**: ESPECIALISTA (especialidade original: oncologia) | **Previsto**: ESPECIALISTA | p(ESPECIALISTA)=0.575
  > Removal of radiation-induced cataracts in patients treated for retinoblastoma. Experience with removal of radiation-induced cataract in patients treated for retinoblastoma is limited. We retrospectively reviewed the reco...

- **Real**: ESPECIALISTA (especialidade original: cardiologia) | **Previsto**: ESPECIALISTA | p(ESPECIALISTA)=0.591
  > Endothelin family in human plasma and cerebrospinal fluid. To clarify whether endothelin may be present in human cerebrospinal fluid (CSF) and, if it exists, to compare its molecular forms with those of endothelin in hum...

- **Real**: CLINICO_GERAL (especialidade original: clinica_geral) | **Previsto**: CLINICO_GERAL | p(ESPECIALISTA)=0.365
  > Synergistic effects of INTERCEED(TC7) and heparin in reducing adhesion formation in the rabbit uterine horn model. Surgical adjuvants are commonly employed to reduce the frequency of postoperative adhesion development af...


## Falsos negativos (real=ESPECIALISTA, previsto=CLINICO_GERAL) — 71 de 1323 no teste

Impacto clínico: paciente que precisava de especialista é encaminhado ao clínico geral — risco de atraso diagnóstico/terapêutico. É o erro mais custoso neste contexto.

- especialidade original: neurologia | p(ESPECIALISTA)=0.393
  > Sensorineural hearing loss: a reversible effect of valproic acid. We report 2 patients over the age of 70 who, while on valproate (VPA) for complex partial seizures, developed sensorineural hearing loss. Following discon...

- especialidade original: neurologia | p(ESPECIALISTA)=0.296
  > Saphenous nerve entrapment caused by pes anserine bursitis mimicking stress fracture of the tibia. Numerous studies have addressed saphenous nerve entrapment at the level of the adductor canal. In this case, we report an...

- especialidade original: cardiologia | p(ESPECIALISTA)=0.363
  > Increased incidence of respiratory distress syndrome in babies of hypertensive mothers. There is controversy over the effect of hypertension in pregnancy on the incidence of neonatal respiratory distress syndrome. We inv...

- especialidade original: gastroenterologia | p(ESPECIALISTA)=0.275
  > Serum gastrin and blood glucose levels during halothane-nitrous oxide anaesthesia and strabismus surgery in children. The purpose of this study was to determine whether serum gastrin levels are increased by reflexogenic ...

- especialidade original: gastroenterologia | p(ESPECIALISTA)=0.345
  > Management of an extensive tracheoesophageal fistula by cervical esophageal exclusion. Giant tracheoesophageal fistulae occurring in ventilator-dependent patients usually result in significant ventilatory embarrassment. ...


## Falsos positivos (real=CLINICO_GERAL, previsto=ESPECIALISTA) — 151 de 1323 no teste

Impacto: paciente sem necessidade clara de especialista é encaminhado a um — custo de recurso/tempo de especialista, mas sem risco direto ao paciente.

- p(ESPECIALISTA)=0.547
  > Persistent primitive trigeminal artery-cavernous sinus fistulas: report of two cases. Two cases of persistent primitive trigeminal artery-cavernous sinus fistulas are presented. In one case, the fistula was treated by us...

- p(ESPECIALISTA)=0.502
  > Clinicopathologic review of twelve children with nephropathy, Wilms tumor, and genital abnormalities (Drash syndrome). The clinicopathologic and radiologic features of 12 children with complete and incomplete forms of Dr...

- p(ESPECIALISTA)=0.520
  > The neural substrate of memory impairment demonstrated by the intracarotid amobarbital procedure. The intracarotid amobarbital sodium (Amytal) procedure (IAP) was performed for 46 patients with temporal lobe epilepsy (21...

- p(ESPECIALISTA)=0.599
  > Memory-contingent saccades and the substantia nigra postulate for essential blepharospasm. Essential blepharospasm and cranial dystonia are related focal dystonias of unknown aetiology. Blepharospasm induced by acute dop...

- p(ESPECIALISTA)=0.576
  > Measurement of D-dimer in plasma as diagnostic aid in suspected pulmonary embolism The potential of plasma measurement of D-dimer (DD), a specific derivative of crosslinked fibrin, for diagnosis or exclusion of pulmonary...
