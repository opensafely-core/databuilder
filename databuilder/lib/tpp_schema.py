import random

import sqlalchemy.orm
from sqlalchemy import Column, Date, ForeignKey, Integer, String

Base = sqlalchemy.orm.declarative_base()
rand = random.Random(12345)


class Patient(Base):
    __tablename__ = "Patient"
    Patient_ID = Column(Integer, primary_key=True)
    Sex = Column(String())
    DateOfBirth = Column(Date)
    DateOfDeath = Column(Date)


def patient(patient_id=None, sex=None, dob=None, date_of_death=None, related=None):
    if not patient_id:
        patient_id = rand.randint(1, 10**6)
    if not related:
        related = []

    for entity in related:
        entity.Patient_ID = patient_id
    return [
        Patient(
            Patient_ID=patient_id, Sex=sex, DateOfBirth=dob, DateOfDeath=date_of_death
        ),
        *related,
    ]


class APCS(Base):
    __tablename__ = "APCS"
    APCS_Ident = Column(Integer, primary_key=True)
    Patient_ID = Column(Integer, ForeignKey("Patient.Patient_ID"))
    Admission_Date = Column(Date)
    Der_Diagnosis_All = Column(String)


def apcs(admission_date=None, codes=None):
    return APCS(
        Admission_Date=(admission_date or "2012-12-12"),
        Der_Diagnosis_All=codes or "xyz",
    )
