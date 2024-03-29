#from pydantic import BaseModel, Field
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class ProviderData(BaseModel):
    provider_id: str = Field(description="Provider ID of the Provider")
    first_name: str = Field(description="First name of the Provider")
    last_name: str = Field(description="Last name of the Provider")
    office_location: str = Field(description="Office location of the Provider")
