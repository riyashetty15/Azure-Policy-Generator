"""
Append synthetic Azure Policy training samples for resource types that are
under-represented in the base dataset. Duplicate instructions are skipped.

Usage:
    python generate_extra_data.py
    python generate_extra_data.py --dataset azure_policy_dataset_clean.json
    python generate_extra_data.py --dry-run
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _p(rule_if: Any, display_name: str) -> str:
    obj = {
        "properties": {
            "displayName": display_name,
            "description": display_name,
            "policyRule": {
                "if": rule_if,
                "then": {"effect": "[parameters('effect')]"},
            },
        }
    }
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


EXTRA: List[Dict[str, str]] = [
    # ── Virtual Machines ──────────────────────────────────────────────────────
    {
        "instruction": "Audit virtual machines that do not use managed disks",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachines"},
                    {
                        "field": "Microsoft.Compute/virtualMachines/storageProfile.osDisk.managedDisk.id",
                        "exists": "false",
                    },
                ]
            },
            "Audit virtual machines that do not use managed disks",
        ),
    },
    {
        "instruction": "Require disk encryption on virtual machine OS disks",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachines"},
                    {
                        "field": "Microsoft.Compute/virtualMachines/storageProfile.osDisk.encryptionSettings",
                        "exists": "false",
                    },
                ]
            },
            "Require disk encryption on virtual machine OS disks",
        ),
    },
    {
        "instruction": "Deny virtual machine network interfaces with a public IP address",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Network/networkInterfaces"},
                    {
                        "count": {
                            "field": "Microsoft.Network/networkInterfaces/ipconfigurations[*].publicIpAddress.id",
                        },
                        "greaterOrEquals": 1,
                    },
                ]
            },
            "Deny virtual machine network interfaces with a public IP address",
        ),
    },
    {
        "instruction": "Require virtual machine scale sets to use managed disks",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachineScaleSets"},
                    {
                        "field": "Microsoft.Compute/VirtualMachineScaleSets/virtualMachineProfile.storageProfile.osDisk.managedDisk.storageAccountType",
                        "exists": "false",
                    },
                ]
            },
            "Require virtual machine scale sets to use managed disks",
        ),
    },
    {
        "instruction": "Restrict allowed virtual machine SKUs to approved sizes only",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Compute/virtualMachines"},
                    {
                        "not": {
                            "field": "Microsoft.Compute/virtualMachines/sku.name",
                            "in": "[parameters('listOfAllowedSKUs')]",
                        }
                    },
                ]
            },
            "Restrict allowed virtual machine SKUs to approved sizes only",
        ),
    },
    # ── SQL Server / Database ─────────────────────────────────────────────────
    {
        "instruction": "Require Azure Active Directory administrator to be provisioned on SQL servers",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Sql/servers"},
                    {
                        "count": {
                            "field": "Microsoft.Sql/servers/administrators[*]",
                            "where": {
                                "field": "Microsoft.Sql/servers/administrators[*].administratorType",
                                "equals": "ActiveDirectory",
                            },
                        },
                        "equals": 0,
                    },
                ]
            },
            "Require Azure Active Directory administrator to be provisioned on SQL servers",
        ),
    },
    {
        "instruction": "Block public endpoint access on SQL servers",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Sql/servers"},
                    {
                        "field": "Microsoft.Sql/servers/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public endpoint access on SQL servers",
        ),
    },
    {
        "instruction": "Enforce minimum TLS version 1.2 on SQL servers",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Sql/servers"},
                    {
                        "anyOf": [
                            {"field": "Microsoft.Sql/servers/minimalTlsVersion", "exists": "false"},
                            {"field": "Microsoft.Sql/servers/minimalTlsVersion", "notEquals": "1.2"},
                        ]
                    },
                ]
            },
            "Enforce minimum TLS version 1.2 on SQL servers",
        ),
    },
    {
        "instruction": "Require private endpoint for Azure SQL servers",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Sql/servers"},
                    {
                        "count": {
                            "field": "Microsoft.Sql/servers/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.Sql/servers/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private endpoint for Azure SQL servers",
        ),
    },
    # ── Azure Kubernetes Service ──────────────────────────────────────────────
    {
        "instruction": "Require role-based access control to be enabled on Kubernetes services",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "field": "Microsoft.ContainerService/managedClusters/enableRBAC",
                        "equals": False,
                    },
                ]
            },
            "Require role-based access control to be enabled on Kubernetes services",
        ),
    },
    {
        "instruction": "Require authorized IP ranges to be defined on Kubernetes services",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "field": "Microsoft.ContainerService/managedClusters/apiServerAccessProfile.authorizedIPRanges",
                        "exists": "false",
                    },
                ]
            },
            "Require authorized IP ranges to be defined on Kubernetes services",
        ),
    },
    {
        "instruction": "Audit Kubernetes clusters that do not use Azure CNI network plugin",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "field": "Microsoft.ContainerService/managedClusters/networkProfile.networkPlugin",
                        "notEquals": "azure",
                    },
                ]
            },
            "Audit Kubernetes clusters that do not use Azure CNI network plugin",
        ),
    },
    {
        "instruction": "Require network policy to be enabled on Kubernetes services",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.ContainerService/managedClusters/networkProfile.networkPolicy",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.ContainerService/managedClusters/networkProfile.networkPolicy",
                                "equals": "",
                            },
                        ]
                    },
                ]
            },
            "Require network policy to be enabled on Kubernetes services",
        ),
    },
    {
        "instruction": "Require OS disk encryption at host for Kubernetes cluster node pools",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "count": {
                            "field": "Microsoft.ContainerService/managedClusters/agentPoolProfiles[*]",
                            "where": {
                                "field": "Microsoft.ContainerService/managedClusters/agentPoolProfiles[*].enableEncryptionAtHost",
                                "notEquals": True,
                            },
                        },
                        "greaterOrEquals": 1,
                    },
                ]
            },
            "Require OS disk encryption at host for Kubernetes cluster node pools",
        ),
    },
    # ── Azure Key Vault ───────────────────────────────────────────────────────
    {
        "instruction": "Require Key Vault soft delete to be enabled",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.KeyVault/vaults"},
                    {
                        "field": "Microsoft.KeyVault/vaults/enableSoftDelete",
                        "notEquals": True,
                    },
                ]
            },
            "Require Key Vault soft delete to be enabled",
        ),
    },
    {
        "instruction": "Require purge protection to be enabled on Azure Key Vault",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.KeyVault/vaults"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.KeyVault/vaults/enablePurgeProtection",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.KeyVault/vaults/enablePurgeProtection",
                                "notEquals": True,
                            },
                        ]
                    },
                ]
            },
            "Require purge protection to be enabled on Azure Key Vault",
        ),
    },
    {
        "instruction": "Block public network access to Azure Key Vault",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.KeyVault/vaults"},
                    {
                        "field": "Microsoft.KeyVault/vaults/networkAcls.defaultAction",
                        "notEquals": "Deny",
                    },
                ]
            },
            "Block public network access to Azure Key Vault",
        ),
    },
    {
        "instruction": "Require Azure Key Vault to use RBAC authorization",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.KeyVault/vaults"},
                    {
                        "field": "Microsoft.KeyVault/vaults/enableRbacAuthorization",
                        "notEquals": True,
                    },
                ]
            },
            "Require Azure Key Vault to use RBAC authorization",
        ),
    },
    {
        "instruction": "Audit Key Vaults that do not have a private endpoint configured",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.KeyVault/vaults"},
                    {
                        "count": {
                            "field": "Microsoft.KeyVault/vaults/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.KeyVault/vaults/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Audit Key Vaults that do not have a private endpoint configured",
        ),
    },
    # ── App Service ───────────────────────────────────────────────────────────
    {
        "instruction": "Require HTTPS only on App Service web apps",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Web/sites"},
                    {"field": "Microsoft.Web/sites/httpsOnly", "notEquals": True},
                ]
            },
            "Require HTTPS only on App Service web apps",
        ),
    },
    {
        "instruction": "Enforce minimum TLS 1.2 on App Service web apps",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Web/sites"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.Web/sites/config/web.minTlsVersion",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.Web/sites/config/web.minTlsVersion",
                                "notEquals": "1.2",
                            },
                        ]
                    },
                ]
            },
            "Enforce minimum TLS 1.2 on App Service web apps",
        ),
    },
    {
        "instruction": "Disable FTP deployments on App Service",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Web/sites"},
                    {
                        "field": "Microsoft.Web/sites/config/web.ftpsState",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Disable FTP deployments on App Service",
        ),
    },
    {
        "instruction": "Require remote debugging to be disabled on App Service",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Web/sites"},
                    {
                        "field": "Microsoft.Web/sites/config/web.remoteDebuggingEnabled",
                        "equals": True,
                    },
                ]
            },
            "Require remote debugging to be disabled on App Service",
        ),
    },
    {
        "instruction": "Require managed identity to be assigned to App Service",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Web/sites"},
                    {
                        "anyOf": [
                            {"field": "identity.type", "exists": "false"},
                            {"field": "identity.type", "equals": "None"},
                        ]
                    },
                ]
            },
            "Require managed identity to be assigned to App Service",
        ),
    },
    # ── Azure Container Registry ──────────────────────────────────────────────
    {
        "instruction": "Disable admin user account on Azure Container Registry",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerRegistry/registries"},
                    {
                        "field": "Microsoft.ContainerRegistry/registries/adminUserEnabled",
                        "equals": True,
                    },
                ]
            },
            "Disable admin user account on Azure Container Registry",
        ),
    },
    {
        "instruction": "Block public network access on Azure Container Registry",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerRegistry/registries"},
                    {
                        "field": "Microsoft.ContainerRegistry/registries/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Container Registry",
        ),
    },
    {
        "instruction": "Require container registries to use private links",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerRegistry/registries"},
                    {
                        "count": {
                            "field": "Microsoft.ContainerRegistry/registries/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.ContainerRegistry/registries/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require container registries to use private links",
        ),
    },
    {
        "instruction": "Require Azure Container Registry to use customer-managed key for encryption",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerRegistry/registries"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.ContainerRegistry/registries/encryption.keyVaultProperties.keyIdentifier",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.ContainerRegistry/registries/encryption.keyVaultProperties.keyIdentifier",
                                "equals": "",
                            },
                        ]
                    },
                ]
            },
            "Require Azure Container Registry to use customer-managed key for encryption",
        ),
    },
    # ── Azure Cosmos DB ───────────────────────────────────────────────────────
    {
        "instruction": "Block public network access on Azure Cosmos DB accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DocumentDB/databaseAccounts"},
                    {
                        "field": "Microsoft.DocumentDB/databaseAccounts/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Cosmos DB accounts",
        ),
    },
    {
        "instruction": "Require private endpoint for Azure Cosmos DB accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DocumentDB/databaseAccounts"},
                    {
                        "count": {
                            "field": "Microsoft.DocumentDB/databaseAccounts/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.DocumentDB/databaseAccounts/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private endpoint for Azure Cosmos DB accounts",
        ),
    },
    {
        "instruction": "Require Cosmos DB accounts to use customer-managed keys to encrypt data at rest",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DocumentDB/databaseAccounts"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.DocumentDB/databaseAccounts/keyVaultKeyUri",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.DocumentDB/databaseAccounts/keyVaultKeyUri",
                                "equals": "",
                            },
                        ]
                    },
                ]
            },
            "Require Cosmos DB accounts to use customer-managed keys to encrypt data at rest",
        ),
    },
    {
        "instruction": "Require automatic failover to be enabled on Azure Cosmos DB accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DocumentDB/databaseAccounts"},
                    {
                        "field": "Microsoft.DocumentDB/databaseAccounts/enableAutomaticFailover",
                        "notEquals": True,
                    },
                ]
            },
            "Require automatic failover to be enabled on Azure Cosmos DB accounts",
        ),
    },
    # ── Event Hub ─────────────────────────────────────────────────────────────
    {
        "instruction": "Enforce minimum TLS 1.2 for Event Hub namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.EventHub/namespaces"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.EventHub/namespaces/minimumTlsVersion",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.EventHub/namespaces/minimumTlsVersion",
                                "notEquals": "1.2",
                            },
                        ]
                    },
                ]
            },
            "Enforce minimum TLS 1.2 for Event Hub namespaces",
        ),
    },
    {
        "instruction": "Block public network access on Event Hub namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.EventHub/namespaces"},
                    {
                        "field": "Microsoft.EventHub/namespaces/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Event Hub namespaces",
        ),
    },
    {
        "instruction": "Require private link for Event Hub namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.EventHub/namespaces"},
                    {
                        "count": {
                            "field": "Microsoft.EventHub/namespaces/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.EventHub/namespaces/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private link for Event Hub namespaces",
        ),
    },
    # ── Service Bus ───────────────────────────────────────────────────────────
    {
        "instruction": "Enforce minimum TLS 1.2 for Service Bus namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ServiceBus/namespaces"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.ServiceBus/namespaces/minimumTlsVersion",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.ServiceBus/namespaces/minimumTlsVersion",
                                "notEquals": "1.2",
                            },
                        ]
                    },
                ]
            },
            "Enforce minimum TLS 1.2 for Service Bus namespaces",
        ),
    },
    {
        "instruction": "Block public network access on Service Bus namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ServiceBus/namespaces"},
                    {
                        "field": "Microsoft.ServiceBus/namespaces/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Service Bus namespaces",
        ),
    },
    {
        "instruction": "Require private endpoint for Service Bus namespaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ServiceBus/namespaces"},
                    {
                        "count": {
                            "field": "Microsoft.ServiceBus/namespaces/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.ServiceBus/namespaces/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private endpoint for Service Bus namespaces",
        ),
    },
    # ── Network Security Groups & VNet ────────────────────────────────────────
    {
        "instruction": "Block inbound RDP access from the internet in network security groups",
        "target": _p(
            {
                "allOf": [
                    {
                        "field": "type",
                        "equals": "Microsoft.Network/networkSecurityGroups/securityRules",
                    },
                    {
                        "allOf": [
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/access",
                                "equals": "Allow",
                            },
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/direction",
                                "equals": "Inbound",
                            },
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/destinationPortRange",
                                "equals": "3389",
                            },
                            {
                                "anyOf": [
                                    {
                                        "field": "Microsoft.Network/networkSecurityGroups/securityRules/sourceAddressPrefix",
                                        "equals": "*",
                                    },
                                    {
                                        "field": "Microsoft.Network/networkSecurityGroups/securityRules/sourceAddressPrefix",
                                        "equals": "Internet",
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            "Block inbound RDP access from the internet in network security groups",
        ),
    },
    {
        "instruction": "Block inbound SSH access from the internet in network security groups",
        "target": _p(
            {
                "allOf": [
                    {
                        "field": "type",
                        "equals": "Microsoft.Network/networkSecurityGroups/securityRules",
                    },
                    {
                        "allOf": [
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/access",
                                "equals": "Allow",
                            },
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/direction",
                                "equals": "Inbound",
                            },
                            {
                                "field": "Microsoft.Network/networkSecurityGroups/securityRules/destinationPortRange",
                                "equals": "22",
                            },
                            {
                                "anyOf": [
                                    {
                                        "field": "Microsoft.Network/networkSecurityGroups/securityRules/sourceAddressPrefix",
                                        "equals": "*",
                                    },
                                    {
                                        "field": "Microsoft.Network/networkSecurityGroups/securityRules/sourceAddressPrefix",
                                        "equals": "Internet",
                                    },
                                ]
                            },
                        ]
                    },
                ]
            },
            "Block inbound SSH access from the internet in network security groups",
        ),
    },
    {
        "instruction": "Require DDoS protection standard to be enabled on virtual networks",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Network/virtualNetworks"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.Network/virtualNetworks/ddosProtectionPlan.id",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.Network/virtualNetworks/enableDdosProtection",
                                "notEquals": True,
                            },
                        ]
                    },
                ]
            },
            "Require DDoS protection standard to be enabled on virtual networks",
        ),
    },
    {
        "instruction": "Require network security groups to be associated with all subnets",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Network/virtualNetworks/subnets"},
                    {
                        "field": "name",
                        "notIn": ["GatewaySubnet", "AzureFirewallSubnet", "AzureBastionSubnet"],
                    },
                    {
                        "field": "Microsoft.Network/virtualNetworks/subnets/networkSecurityGroup.id",
                        "exists": "false",
                    },
                ]
            },
            "Require network security groups to be associated with all subnets",
        ),
    },
    # ── Cognitive Services / Azure OpenAI ─────────────────────────────────────
    {
        "instruction": "Block public network access on Cognitive Services accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.CognitiveServices/accounts"},
                    {
                        "field": "Microsoft.CognitiveServices/accounts/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Cognitive Services accounts",
        ),
    },
    {
        "instruction": "Require Cognitive Services accounts to use a customer-managed key for encryption",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.CognitiveServices/accounts"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.CognitiveServices/accounts/encryption.keySource",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.CognitiveServices/accounts/encryption.keySource",
                                "notEquals": "Microsoft.KeyVault",
                            },
                        ]
                    },
                ]
            },
            "Require Cognitive Services accounts to use a customer-managed key for encryption",
        ),
    },
    {
        "instruction": "Require managed identity on Cognitive Services accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.CognitiveServices/accounts"},
                    {
                        "anyOf": [
                            {"field": "identity.type", "exists": "false"},
                            {"field": "identity.type", "equals": "None"},
                        ]
                    },
                ]
            },
            "Require managed identity on Cognitive Services accounts",
        ),
    },
    {
        "instruction": "Block public network access on Azure OpenAI accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.CognitiveServices/accounts"},
                    {"field": "kind", "equals": "OpenAI"},
                    {
                        "field": "Microsoft.CognitiveServices/accounts/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure OpenAI accounts",
        ),
    },
    # ── Azure Cache for Redis ─────────────────────────────────────────────────
    {
        "instruction": "Enforce minimum TLS 1.2 for Azure Cache for Redis",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Cache/Redis"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.Cache/Redis/minimumTlsVersion",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.Cache/Redis/minimumTlsVersion",
                                "notEquals": "1.2",
                            },
                        ]
                    },
                ]
            },
            "Enforce minimum TLS 1.2 for Azure Cache for Redis",
        ),
    },
    {
        "instruction": "Disable non-SSL port on Azure Cache for Redis",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Cache/Redis"},
                    {"field": "Microsoft.Cache/Redis/enableNonSslPort", "equals": True},
                ]
            },
            "Disable non-SSL port on Azure Cache for Redis",
        ),
    },
    {
        "instruction": "Require private link for Azure Cache for Redis",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Cache/Redis"},
                    {
                        "count": {
                            "field": "Microsoft.Cache/Redis/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.Cache/Redis/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private link for Azure Cache for Redis",
        ),
    },
    # ── Azure Machine Learning ────────────────────────────────────────────────
    {
        "instruction": "Block public network access on Azure Machine Learning workspaces",
        "target": _p(
            {
                "allOf": [
                    {
                        "field": "type",
                        "equals": "Microsoft.MachineLearningServices/workspaces",
                    },
                    {
                        "field": "Microsoft.MachineLearningServices/workspaces/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Machine Learning workspaces",
        ),
    },
    {
        "instruction": "Require Azure Machine Learning workspaces to use customer-managed keys",
        "target": _p(
            {
                "allOf": [
                    {
                        "field": "type",
                        "equals": "Microsoft.MachineLearningServices/workspaces",
                    },
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.MachineLearningServices/workspaces/encryption.keyVaultProperties.keyIdentifier",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.MachineLearningServices/workspaces/encryption.keyVaultProperties.keyIdentifier",
                                "equals": "",
                            },
                        ]
                    },
                ]
            },
            "Require Azure Machine Learning workspaces to use customer-managed keys",
        ),
    },
    # ── Azure Batch ───────────────────────────────────────────────────────────
    {
        "instruction": "Require Azure Batch accounts to use customer-managed keys for data encryption",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Batch/batchAccounts"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.Batch/batchAccounts/encryption.keySource",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.Batch/batchAccounts/encryption.keySource",
                                "notEquals": "Microsoft.KeyVault",
                            },
                        ]
                    },
                ]
            },
            "Require Azure Batch accounts to use customer-managed keys for data encryption",
        ),
    },
    {
        "instruction": "Block public network access on Azure Batch accounts",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Batch/batchAccounts"},
                    {
                        "field": "Microsoft.Batch/batchAccounts/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Batch accounts",
        ),
    },
    # ── Azure Data Factory ────────────────────────────────────────────────────
    {
        "instruction": "Block public network access on Azure Data Factory",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DataFactory/factories"},
                    {
                        "field": "Microsoft.DataFactory/factories/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Data Factory",
        ),
    },
    {
        "instruction": "Require private link for Azure Data Factory",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.DataFactory/factories"},
                    {
                        "count": {
                            "field": "Microsoft.DataFactory/factories/privateEndpointConnections[*]",
                            "where": {
                                "field": "Microsoft.DataFactory/factories/privateEndpointConnections[*].privateLinkServiceConnectionState.status",
                                "equals": "Approved",
                            },
                        },
                        "less": 1,
                    },
                ]
            },
            "Require private link for Azure Data Factory",
        ),
    },
    # ── Azure Databricks ──────────────────────────────────────────────────────
    {
        "instruction": "Block public network access on Azure Databricks workspaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Databricks/workspaces"},
                    {
                        "field": "Microsoft.Databricks/workspaces/parameters.enableNoPublicIp.value",
                        "notEquals": True,
                    },
                ]
            },
            "Block public network access on Azure Databricks workspaces",
        ),
    },
    # ── Tags ──────────────────────────────────────────────────────────────────
    {
        "instruction": "Require a cost-center tag on all resources",
        "target": _p(
            {"allOf": [{"field": "tags['cost-center']", "exists": "false"}]},
            "Require a cost-center tag on all resources",
        ),
    },
    {
        "instruction": "Require an environment tag on all resource groups",
        "target": _p(
            {
                "allOf": [
                    {
                        "field": "type",
                        "equals": "Microsoft.Resources/subscriptions/resourceGroups",
                    },
                    {"field": "tags['environment']", "exists": "false"},
                ]
            },
            "Require an environment tag on all resource groups",
        ),
    },
    {
        "instruction": "Deny creation of resources without a department tag",
        "target": _p(
            {"allOf": [{"field": "tags['department']", "exists": "false"}]},
            "Deny creation of resources without a department tag",
        ),
    },
    {
        "instruction": "Require a project tag on all resources",
        "target": _p(
            {"allOf": [{"field": "tags['project']", "exists": "false"}]},
            "Require a project tag on all resources",
        ),
    },
    # ── Azure Kubernetes Service - additional ─────────────────────────────────
    {
        "instruction": "Require Azure Defender for Kubernetes to be enabled",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerService/managedClusters"},
                    {
                        "field": "Microsoft.ContainerService/managedClusters/securityProfile.defender.securityMonitoring.enabled",
                        "notEquals": True,
                    },
                ]
            },
            "Require Azure Defender for Kubernetes to be enabled",
        ),
    },
    # ── Azure Container Instances ─────────────────────────────────────────────
    {
        "instruction": "Require container groups to use a virtual network",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.ContainerInstance/containerGroups"},
                    {
                        "field": "Microsoft.ContainerInstance/containerGroups/subnetIds[*].id",
                        "exists": "false",
                    },
                ]
            },
            "Require container groups to use a virtual network",
        ),
    },
    # ── Azure Synapse Analytics ───────────────────────────────────────────────
    {
        "instruction": "Block public network access on Azure Synapse workspaces",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Synapse/workspaces"},
                    {
                        "field": "Microsoft.Synapse/workspaces/publicNetworkAccess",
                        "notEquals": "Disabled",
                    },
                ]
            },
            "Block public network access on Azure Synapse workspaces",
        ),
    },
    {
        "instruction": "Require Azure Synapse workspaces to use customer-managed keys",
        "target": _p(
            {
                "allOf": [
                    {"field": "type", "equals": "Microsoft.Synapse/workspaces"},
                    {
                        "anyOf": [
                            {
                                "field": "Microsoft.Synapse/workspaces/encryption.cmk.key.keyVaultUrl",
                                "exists": "false",
                            },
                            {
                                "field": "Microsoft.Synapse/workspaces/encryption.cmk.key.keyVaultUrl",
                                "equals": "",
                            },
                        ]
                    },
                ]
            },
            "Require Azure Synapse workspaces to use customer-managed keys",
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append extra Azure Policy training samples to dataset."
    )
    parser.add_argument(
        "--dataset",
        default="azure_policy_dataset_clean.json",
        help="Dataset JSONL file to extend (appended in place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be added without writing",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)

    existing_instructions: set = set()
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    instr = (row.get("instruction") or "").strip().lower()
                    if instr:
                        existing_instructions.add(instr)
                except json.JSONDecodeError:
                    continue

    new_samples = []
    skipped = 0
    for sample in EXTRA:
        key = sample["instruction"].strip().lower()
        if key in existing_instructions:
            skipped += 1
        else:
            new_samples.append(sample)
            existing_instructions.add(key)

    if args.dry_run:
        for s in new_samples:
            print(json.dumps(s, ensure_ascii=False))
        print(f"\nWould add {len(new_samples)} samples, skip {skipped} duplicates.")
        return

    with dataset_path.open("ab") as f:
        # Ensure the file ends with a newline before appending.
        if dataset_path.stat().st_size > 0:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 1))
            if f.read(1) != b"\n":
                f.write(b"\n")
        for s in new_samples:
            f.write((json.dumps(s, ensure_ascii=False) + "\n").encode("utf-8"))

    print(f"Added {len(new_samples)} new samples to {dataset_path} (skipped {skipped} duplicates).")


if __name__ == "__main__":
    main()
