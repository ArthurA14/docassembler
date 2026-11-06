{{- define "projectname.name" -}}
{{- default .Chart.Name .Values.metadata.nameOverride -}}
{{- end }}

{{- define "projectname.fullname" -}}
{{- if .Values.metadata.fullnameOverride -}}
{{- .Values.metadata.fullnameOverride -}}
{{- else -}}
{{- printf "%s-%s" (include "projectname.name" .) .Release.Name -}}
{{- end -}}
{{- end }}

{{- define "projectname.namespace" -}}
{{- default .Release.Namespace .Values.metadata.namespaceOverride -}}
{{- end }}

{{- define "projectname.component" -}}
{{- default "platform" .Values.metadata.component | quote -}}
{{- end }}

{{/*
define "projectname.labels"
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/part-of: {{ include "projectname.fullname" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}  # Helm
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
end
*/}}
