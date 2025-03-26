from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from apis.model_ai.model_store import get_model
from apis.routers.models import SentenceInput
from apis.routers.utils import predict_hscode

router=APIRouter()

@router.get("/ping_to_server")
def ping_to_server():
    return {"message":"This is server classify hscode"}


@router.post("/single_predict")
async def sentence_single(input_data: SentenceInput):
    try:
        retrieval_model = get_model()
        if retrieval_model is None:
            raise HTTPException(
                status_code=500, detail="Could not load model retriever")

        results = await predict_hscode(input_data, retrieval_model)

        return JSONResponse(content={"status_code": status.HTTP_200_OK, "data": results})

    except Exception as e:
        print(f"Error in predict_each_sentence: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
