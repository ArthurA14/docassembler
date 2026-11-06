from fastapi import Request


def get_document_manager(request: Request):
    return request.app.state.document_manager