package core

import "strings"

func (m *SIPMessage) Method() string {
	fields := strings.Fields(m.StartLine)

	if len(fields) == 0 {
		return ""
	}

	return fields[0]
}
