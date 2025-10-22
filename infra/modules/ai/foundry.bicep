param location string
param suffix string
param chatCompletionModel string
param embeddingModel string

var aiFoundryName = 'aifoundry${suffix}'

#disable-next-line BCP036
resource aiFoundry 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: aiFoundryName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  kind: 'AIServices'
  properties: {
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
    // required to work in AI Foundry
    allowProjectManagement: true
    // true is not supported today
    disableLocalAuth: false
    customSubDomainName: aiFoundryName
    networkInjections: null
  }
}

resource chatModelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundry
  name: chatCompletionModel
  sku: {
    capacity: 1
    name: 'GlobalStandard'
  }
  properties: {
    model: {
      name: chatCompletionModel
      format: 'OpenAI'
    }
  }
}

resource embeddingModelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: aiFoundry
  dependsOn: [
    chatModelDeployment
  ]
  name: embeddingModel
  sku: {
    capacity: 1
    name: 'GlobalStandard'
  }
  properties: {
    model: {
      name: embeddingModel
      format: 'OpenAI'
    }
  }
}

output resourceId string = aiFoundry.id
output resourceName string = aiFoundry.name
output systemAssignedMIPrincipalId string = aiFoundry.identity.principalId
