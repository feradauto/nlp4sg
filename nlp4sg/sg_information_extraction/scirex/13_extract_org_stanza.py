import time
import pandas as pd
import numpy as np
import stanza

nlp = stanza.Pipeline('en', processors={'ner': 'OntoNotes'})

def get_org(x):
    doc=nlp(x)
    orgs=[]
    for sentence in doc.sentences:
        for e in sentence.ents:
            if e.type=='ORG':
                orgs.append(e.text)
    return orgs

def get_organizations(positives):
    """
    Use stanza to parse organizations from the acknowledgments and if that section doesn't exist, get them from
    the abstract restricting to organizations that are common in the acknowledgments
    """
    positives=positives.assign(abstract_for_prompt=positives.abstract_for_prompt.fillna(""))
    positives=positives.assign(acknowledgments_for_prompt=positives.acknowledgments_for_prompt.fillna(""))


    positives=positives.assign(organization_ack=positives.acknowledgments_for_prompt.apply(lambda x:get_org(x)))

    positives=positives.assign(organization_abstract=positives.title_abstract_clean.apply(lambda x:get_org(x)))
    ## Acknowledgments
    orgs_ack=positives.loc[:,['ID','organization_ack']]
    orgs_ack=orgs_ack.explode('organization_ack')
    orgs_ack=orgs_ack.dropna()

    # abstract
    orgs_abs=positives.loc[:,['ID','organization_abstract']]
    orgs_abs=orgs_abs.explode('organization_abstract')
    orgs_abs=orgs_abs.dropna()
    ## restrict the ones extracted from abstract
    orgs_ack_common=orgs_ack.organization_ack.value_counts().reset_index().rename(columns={'index':'name'}).head(400)
    orgs_ack_common=orgs_ack_common.loc[~orgs_ack_common.name.isin(['No organization','Twitter', 'NLP', 'Reddit', 'Facebook',
           'Social Media', 'EU', 'ASR', 'BioNLP', 'AI', 'BioASQ', 'Google','IBM'
           'CLARIN', 'Amazon', 'ACL', 'ERC', 'Microsoft', 'SemEval'])]
    orgs_abs_real=orgs_abs.loc[orgs_abs.organization_abstract.isin(orgs_ack_common.name.unique())]

    orgs_ack=orgs_ack.rename(columns={'organization_ack':'organization'})
    orgs_abs_real=orgs_abs_real.rename(columns={'organization_abstract':'organization'})
    # concat them
    organizations=pd.concat([orgs_ack,orgs_abs_real])
    ## hand crafted rules
    organizations=organizations.loc[~organizations.organization.isin(['NLP'])]
    organizations=organizations.assign(organization=organizations.organization.str.lstrip("the "))
    return organizations

def main():
    data_path="../../data/"
    output_path="../../outputs/"

    positives=pd.read_csv(output_path+"sg_ie/positives_ready.csv")
    organizations=get_organizations(positives)
    organizations.to_csv(output_path+"sg_ie/organizations_stanza_ontonotes_final.csv",index=False)
    
    test=pd.read_csv(output_path+"sg_ie/test_ready.csv")
    organizations_test=get_organizations(test)
    organizations_test.to_csv(output_path+"sg_ie/organizations_test_stanza_ontonotes_final.csv",index=False)
    
if __name__ == '__main__':
    main()

