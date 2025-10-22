targetScope = 'subscription'

@description('The location where all resources will be created')
@allowed([
  'eastus2'
  'westus3'
  'westus'
  'canadaeast'
])
param location string

@description('The name of the resource group')
param resourceGroupName string

@description('Suffix for the resource group')
param suffix string

@description('The chat completion model to deploy, be sure its supported in the specific region')
@allowed([
  'gpt-4o-mini'
  'gpt-4.1-mini'
  'gpt-4o'
])
param chatCompletionModel string

@description('The embedding model to deploy, be sure its supported in the specific region')
@allowed([
  'text-embedding-ada-002'
  'text-embedding-3-small'
  'text-embedding-3-large'
])
param embeddingModel string

/* Create the resource group */
resource rg 'Microsoft.Resources/resourceGroups@2025-04-01' = {
  name: resourceGroupName
  location: location
}

module search 'br/public:avm/res/search/search-service:0.7.2' = {
  scope: rg
  params: {
    disableLocalAuth: false
    authOptions: {
      aadOrApiKey: { aadAuthFailureMode: 'http401WithBearerChallenge' }
    }
    publicNetworkAccess: 'Enabled'
    name: 'search${replace(suffix,'-','')}'
    location: location
    managedIdentities: {
      systemAssigned: true
    }
    partitionCount: 1
    replicaCount: 1
    sku: 'standard'
  }
}

module foundry 'modules/ai/foundry.bicep' = {
  scope: rg
  params: {
    location: location
    chatCompletionModel: chatCompletionModel
    embeddingModel: embeddingModel
    suffix: suffix
  }
}
