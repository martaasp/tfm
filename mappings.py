### Historical data
admission_type_id_mapping = {'1': 'Emergency','2':'Urgent', '3':'Elective', '4':'Newborn', '5':'Not Available', 
                             '6':'NULL', '7':'Trauma Center', '8':'Not Mapped'}
discharge_disposition_id_mapping = {
    '1':'Discharged to home',
    '2':'Discharged/transferred to another short term hospital',
    '3':'Discharged/transferred to SNF',
    '4':'Discharged/transferred to ICF',
    '5':'Discharged/transferred to another type of inpatient care institution',
    '6':'Discharged/transferred to home with home health service',
    '7':'Left AMA',
    '8':'Discharged/transferred to home under care of Home IV provider',
    '9':'Admitted as an inpatient to this hospital',
    '10':'Neonate discharged to another hospital for neonatal aftercare',
    '11':'Expired',
    '12':'Still patient or expected to return for outpatient services',
    '13':'Hospice / home',
    '14':'Hospice / medical facility',
    '15':'Discharged/transferred within this institution to Medicare approved swing bed',
    '16':'Discharged/transferred/referred another institution for outpatient services',
    '17':'Discharged/transferred/referred to this institution for outpatient services',
    '18':'NULL',
    '19':'Expired at home. Medicaid only, hospice.',
    '20':"Expired in a medical facility. Medicaid only, hospice.",
    '21':"Expired, place unknown. Medicaid only, hospice.",
    '22':'Discharged/transferred to another rehab fac including rehab units of a hospital.',
    '23':'Discharged/transferred to a long term care hospital.',
    '24':'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
    '25':'Not Mapped',
    '26':'Unknown/Invalid',
    '30':'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
    '27':'Discharged/transferred to a federal health care facility.',
    '28':'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    '29':'Discharged/transferred to a Critical Access Hospital (CAH).'
    }
admission_source_id_mapping = {
    '1':'Physician Referral',
    '2':'Clinic Referral',
    '3':'HMO Referral',
    '4':'Transfer from a hospital',
    '5':'Transfer from a Skilled Nursing Facility (SNF)',
    '6':'Transfer from another health care facility',
    '7':'Emergency Room',
    '8':'Court/Law Enforcement',
    '9':'Not Available',
    '10':'Transfer from critial access hospital',
    '11':'Normal Delivery',
    '12':'Premature Delivery',
    '13':'Sick Baby',
    '14':'Extramural Birth',
    '15':'Not Available',
    '17':'NULL',
    '18':'Transfer From Another Home Health Agency',
    '19':'Readmission to Same Home Health Agency',
    '20':'Not Mapped',
    '21':'Unknown/Invalid',
    '22':'Transfer from hospital inpt/same fac reslt in a sep claim',
    '23':'Born inside this hospital',
    '24':'Born outside this hospital',
    '25':'Transfer from Ambulatory Surgery Center',
    '26':'Transfer from Hospice'
    }

meds_mapping = {'Up': 1, 'Down': 2, 'Steady': 3, 'No': 4}

change_mapping = {"No": 0, "Ch" : 1}
diabetesMed_mapping = {"No": 0, "Yes" : 1}
readmitted_mapping = {"<30": 0, ">30" : 0, "NO": 1}


### Prediction data
mapping = {
    '33' : 'Regular insulin dose',
    '34' : 'NPH insulin dose',
    '35' : 'UltraLente insulin dose',
    '48' : 'Unspecified blood glucose measurement',
    '57' : 'Unspecified blood glucose measurement',
    '58' : 'Pre-breakfast blood glucose measurement',
    '59' : 'Post-breakfast blood glucose measurement',
    '60' : 'Pre-lunch blood glucose measurement',
    '61' : 'Post-lunch blood glucose measurement',
    '62' : 'Pre-supper blood glucose measurement',
    '63' : 'Post-supper blood glucose measurement',
    '64' : 'Pre-snack blood glucose measurement',
    '65' : 'Hypoglycemic symptoms',
    '66' : 'Typical meal ingestion',
    '67' : 'More-than-usual meal ingestion',
    '68' : 'Less-than-usual meal ingestion',
    '69' : 'Typical exercise activity',
    '70' : 'More-than-usual exercise activity',
    '71' : 'Less-than-usual exercise activity',
    '72' : 'Unspecified special event'}
