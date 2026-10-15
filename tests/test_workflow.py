"""Small contract tests using invented records; no competition data or model download."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocess import clean_chief_complaint, preprocess_text_series
from src.evaluate import evaluate_acuity
from src.data_io import load_train_test
from src.validation import validate_submission
from src.features_text import fit_transform_tfidf
from src.features_structured import infer_structured_columns, fit_transform_structured
from src.utils import make_classifier


class WorkflowTests(unittest.TestCase):
    def test_text_missing_and_abbreviations(self):
        self.assertEqual(preprocess_text_series([None, np.nan, pd.NA, '']), ['', '', '', ''])
        self.assertEqual(clean_chief_complaint('UTI + bp!'), 'urinary tract infection blood pressure')
        self.assertEqual(clean_chief_complaint('stimulation'), 'stimulation')

    def test_acuity_direction(self):
        m = evaluate_acuity([1, 2, 3, 4, 5], [3, 1, 3, 5, 5])
        self.assertEqual(m['undertriage_rate'], 0.4)
        self.assertEqual(m['severe_undertriage_rate'], 0.2)
        self.assertEqual(m['recall_esi_1_2'], 0.5)
        with self.assertRaises(ValueError): evaluate_acuity([0, 1], [1, 2])

    def test_submission_contract(self):
        f = pd.DataFrame({'patient_id': ['a','b'], 'triage_acuity': [1,5]})
        validate_submission(f, pd.Series(['a','b']))
        with self.assertRaises(ValueError): validate_submission(f, pd.Series(['b','a']))
        with self.assertRaises(ValueError): validate_submission(f.assign(triage_acuity=[1,6]), pd.Series(['a','b']))

    def test_data_join_conflicts_missing_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            pd.DataFrame({'patient_id':['b','a'], 'triage_acuity':[1,2]}).to_csv(p/'train.csv',index=False)
            pd.DataFrame({'patient_id':['c']}).to_csv(p/'test.csv',index=False)
            pd.DataFrame({'patient_id':['a','b','c'], 'chief_complaint_raw':['cough',None,'UTI']}).to_csv(p/'chief_complaints.csv',index=False)
            cfg={'data':{'local_data_dir':d,'train_file':'train.csv','test_file':'test.csv','id_col':'patient_id','target_col':'triage_acuity','include_patient_history':False}}
            tr,te=load_train_test(cfg)
            self.assertEqual(tr.patient_id.tolist(),['b','a'])
            self.assertEqual(tr.chief_complaint_raw.tolist(),['','cough'])
            pd.DataFrame({'patient_id':['a','a'], 'chief_complaint_raw':['cough','fever']}).to_csv(p/'chief_complaints.csv',index=False)
            with self.assertRaises(ValueError): load_train_test(cfg)
            (p/'chief_complaints.csv').unlink()
            with self.assertRaises(FileNotFoundError): load_train_test(cfg)

    def test_export_validates_ids_and_writes_expected_file(self):
        from unittest.mock import patch
        from src.inference import export_submission_from_artifact
        test = pd.DataFrame({'patient_id': ['b', 'a']})
        cfg = {'data': {'id_col': 'patient_id', 'target_col': 'triage_acuity'}}
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'predictions.csv'
            with patch('src.inference.load_train_test', return_value=(None, test)), patch(
                'src.inference.predict_from_artifact', return_value=np.array([1, 5])):
                export_submission_from_artifact(cfg, 'unused.joblib', output)
            written = pd.read_csv(output)
            self.assertEqual(written.patient_id.tolist(), ['b', 'a'])
            self.assertEqual(written.triage_acuity.tolist(), [1, 5])
            with patch('src.inference.load_train_test', return_value=(None, test)), patch(
                'src.inference.predict_from_artifact', return_value=np.array([1, 9])):
                with self.assertRaises(ValueError):
                    export_submission_from_artifact(cfg, 'unused.joblib', output)

    def test_learning_is_fit_on_training_only(self):
        vec,x,y=fit_transform_tfidf(['cough cough','cough fever'],['unseen'])
        self.assertNotIn('unseen',vec.vocabulary_)
        self.assertEqual(y.nnz,0)
        tr=pd.DataFrame({'patient_id':['a','b'],'triage_acuity':[1,2], 'chief_complaint_raw':['a','b'],'age':[10,20],'sex':['F','M'],'disposition':['home','ward']})
        cols=infer_structured_columns(tr,'triage_acuity','chief_complaint_raw','patient_id',['disposition'])
        self.assertEqual(cols,['age','sex'])
        _,a,b=fit_transform_structured(tr[cols],pd.DataFrame({'age':[None],'sex':['new']}))
        self.assertTrue(np.isfinite(b).all())

    def test_small_train_predict_roundtrip(self):
        from sklearn.preprocessing import LabelEncoder
        from src.inference import predict_from_artifact
        import joblib
        texts=['cough','cough','fever','fever','pain','pain']
        vec,x,_=fit_transform_tfidf(texts)
        enc=LabelEncoder().fit(['1','3','5']); labels=enc.transform(['5','5','1','1','3','3'])
        model=make_classifier('logistic_regression',{},3).fit(x,labels)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.joblib'
            joblib.dump({'model':model,'feature_mode':'text_tfidf','text_col':'complaint','tfidf_vectorizer':vec,'label_encoder':enc},p)
            pred=predict_from_artifact(p,pd.DataFrame({'complaint':['fever','cough']}))
            self.assertEqual(pred.tolist(),[1,5])
            from unittest.mock import patch
            from src.inference import export_submission_from_artifact
            records = pd.DataFrame({'patient_id':['x','y'], 'complaint':['fever','cough']})
            cfg = {'data': {'id_col':'patient_id', 'target_col':'triage_acuity'}}
            with patch('src.inference.load_train_test', return_value=(None, records)):
                output = export_submission_from_artifact(cfg, p, Path(d)/'roundtrip.csv')
            self.assertEqual(pd.read_csv(output).triage_acuity.tolist(), [1,5])
        with self.assertRaises(ValueError): make_classifier('typo',{},3)


if __name__=='__main__': unittest.main()
