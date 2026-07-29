from document_service import apply_document_update


def test_apply_document_update_changes_selected_document():
    state = {"settings": {"scan_mode": "mass"}, "documents": []}
    document = {
        "metadata": {
            "lot": "86",
            "address": "2432 COMPTROLLERS CT",
            "tax_map": "0027",
            "parcel": "0456",
            "tax_id": "01-253148",
            "section": "000",
            "project_code": "CC6767",
        }
    }
    payload = {
        "lot": "86",
        "address": "100 JIBSAIL DR",
        "tax_map": "0027",
        "parcel": "0456",
        "tax_id": "01-253148",
        "section": "000",
        "project_code": "CC6767",
        "document_type": "Site Plan",
        "changed_field": "address",
    }

    updated_doc = apply_document_update(state, document, payload)

    assert (
        updated_doc.get("metadata").get("address")  # type: ignore
        == "00100 JIBSAIL DR PRINCE FREDERICK MD 20678"
    )
    assert updated_doc.get("metadata").get("tax_id") == "02-129094"  # type: ignore
    assert updated_doc.get("metadata").get("lot") == "43"  # type: ignore
    assert (
        document.get("metadata").get("address")  # type: ignore
        == "00100 JIBSAIL DR PRINCE FREDERICK MD 20678"
    )
    assert document.get("metadata").get("tax_id") == "02-129094"  # type: ignore
    assert document.get("metadata").get("lot") == "43"  # type: ignore


def test_apply_document_update_batch_syncs_property_fields():
    state = {
        "settings": {"scan_mode": "batch"},
        "documents": [
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
        ],
    }
    document = {
        "metadata": {
            "lot": "86",
            "address": "2432 COMPTROLLERS CT",
            "tax_map": "0027",
            "parcel": "0456",
            "tax_id": "01-253148",
            "section": "000",
            "project_code": "CC6767",
        }
    }
    payload = {
        "lot": "86",
        "address": "2432 COMPTROLLERS CT",
        "tax_map": "0027",
        "parcel": "0456",
        "tax_id": "02-129094",
        "section": "000",
        "project_code": "CC6767",
        "document_type": "Site Plan",
        "changed_field": "tax_id",
    }

    apply_document_update(state, document, payload)
    updated_docs = state.get("documents")

    for doc in updated_docs:  # type: ignore
        assert (
            doc.get("metadata").get("address")  # type: ignore
            == "00100 JIBSAIL DR PRINCE FREDERICK MD 20678"
        )
        assert doc.get("metadata").get("tax_id") == "02-129094"  # type: ignore
        assert doc.get("metadata").get("lot") == "43"  # type: ignore


def test_apply_document_update_mass_does_not_sync_other_documents():
    state = {
        "settings": {"scan_mode": "mass"},
        "documents": [
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
            {
                "metadata": {
                    "lot": "86",
                    "address": "2432 COMPTROLLERS CT",
                    "tax_map": "0027",
                    "parcel": "0456",
                    "tax_id": "01-253148",
                    "section": "000",
                    "project_code": "CC6767",
                }
            },
        ],
    }
    document = {
        "metadata": {
            "lot": "86",
            "address": "2432 COMPTROLLERS CT",
            "tax_map": "0027",
            "parcel": "0456",
            "tax_id": "01-253148",
            "section": "000",
            "project_code": "CC6767",
        }
    }
    payload = {
        "lot": "86",
        "address": "2432 COMPTROLLERS CT",
        "tax_map": "0027",
        "parcel": "0456",
        "tax_id": "02-129094",
        "section": "000",
        "project_code": "CC6767",
        "document_type": "Site Plan",
        "changed_field": "tax_id",
    }

    apply_document_update(state, document, payload)
    updated_docs = state.get("documents")

    for doc in updated_docs:  # type: ignore
        assert (
            doc.get("metadata").get("address")  # type: ignore
            == "2432 COMPTROLLERS CT"
        )
        assert doc.get("metadata").get("tax_id") == "01-253148"  # type: ignore
        assert doc.get("metadata").get("lot") == "86"  # type: ignore
    assert (
            document.get("metadata").get("address")  # type: ignore
            == "00100 JIBSAIL DR PRINCE FREDERICK MD 20678"
        )
    assert document.get("metadata").get("tax_id") == "02-129094"  # type: ignore
    assert document.get("metadata").get("lot") == "43"  # type: ignore
